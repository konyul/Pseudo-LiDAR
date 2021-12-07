import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import build_neck
from . import ddn, ddn_loss
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from kornia.geometry.conversions import convert_points_to_homogeneous
from pcdet.datasets.kitti.kitti_utils import project_image_to_velo, project_image_to_cam

class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )
        if model_cfg.get('CHANNEL_REDUCE',None) is not None:
            self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}
        
        if "PL" in model_cfg:
            self.PL = model_cfg.PL
        else:
            self.PL = False
        if "FS" in model_cfg:
            self.FS = model_cfg.FS
        else:
            self.FS = False

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]

        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)
        
        
        # cv2.imwrite("image.jpg",(np.transpose((images.squeeze(0).cpu().numpy()*255).astype("uint8"),(1,2,0))))
        
        # image_feature = self.normalize(image_features)
        # #sum_image_feature = (np.sum(np.transpose(image_feature,(1,2,0)),axis=2)/64).astype("uint8")
        # max_image_feature = np.max(np.transpose(image_feature.astype("uint8"),(1,2,0)),axis=2)
        # #sum_image_feature = cv2.applyColorMap(sum_image_feature,cv2.COLORMAP_JET)
        # max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
        # #cv2.imwrite("sum_image_feature.jpg",sum_image_feature)
        # cv2.imwrite("max_image_feature.jpg",max_image_feature)
        

        # Create image feature plane-sweep volume
        if self.PL:
            depth_dim, depth_min, depth_max, num_bins = 1, 2, 46.8, 80
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            depth_range = []
            if 'logits_ups' in ddn_result:
                depth_logits_ups = ddn_result['logits_ups']
                h,w = batch_dict['depth_maps'].shape[1], batch_dict['depth_maps'].shape[2]
                depth_logits = F.interpolate(depth_logits.clone(), size=(h, w), mode='bilinear', align_corners=False)
            else:
                depth_logits_ups = F.interpolate(depth_logits, size=(375,1242), mode='bilinear', align_corners=False)
            
            if self.FS:
                if "logits_ones" in ddn_result:
                    depth_logits_ = ddn_result['logits_ones']
                frustum_features = self.create_frustum_features(image_features=image_features,
                                                            depth_logits=depth_logits_)
                batch_dict["frustum_features"] = frustum_features
                
            for indice in range(81):
                depth_range.append((((indice + 0.5) / 0.5)**2 - 1) * bin_size / 8 + depth_min)
            depth_tensor = np.reshape(np.array(depth_range),[1, self.disc_cfg["num_bins"] + 1, 1, 1])
            depth_bin = Variable(torch.Tensor(depth_tensor).cuda(), requires_grad=False)
            depth_bin = depth_bin.repeat(depth_logits_ups.size()[0], 1, depth_logits_ups.size()[2], depth_logits_ups.size()[3])
            
            depth_probs = F.softmax(depth_logits_ups, dim = depth_dim) 
            #depth_probs = depth_logits_ups.clone()     ## e2e
            

            #### sum
            #depth_map_PL = torch.sum(depth_bin*depth_probs,1)
            

            #### max
            dl_m_val = depth_probs.max(dim=1)[0].unsqueeze(1)
            dl_m_idx = depth_probs.max(dim=1)[1].unsqueeze(1)
            mask = torch.zeros_like(depth_probs)
            mask.scatter_(1, dl_m_idx, 1 / dl_m_val)
            depth_probs *= mask
            depth_map_PL = torch.sum(depth_bin*depth_probs,1)
            

            pseudo_lidar = self.depth_to_point(depth_map_PL, lidar_to_cam=batch_dict["trans_lidar_to_cam"],
                                cam_to_img=batch_dict["trans_cam_to_img"],
                                image_shape=batch_dict["image_shape"])  # (B, X, Y, Z, 3)))
            # batch_dict["points"] = list(pseudo_lidar)
            batch_dict["points"] = pseudo_lidar
        else:
            frustum_features = self.create_frustum_features(image_features=image_features,
                                                            depth_logits=depth_logits)
            batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits
        return batch_dict


    def depth_to_point(self, depth_map, lidar_to_cam, cam_to_img, image_shape):       # e2e에서 가져옴 depth_to_pcl
        lst = []
        for i, depth_map_ in enumerate(depth_map):
            rows, cols = depth_map_.shape[0], depth_map_.shape[1]
            c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
            points = torch.stack([c.t(), r.t(), depth_map_], dim=0)
            points = points.reshape((3, -1))
            points = points.t()

            # if torch.sum(points.isnan()).item() != 0:
            #     points[(points.isnan()==True).nonzero(as_tuple=True)[0],2] = 46.8

            cloud = project_image_to_velo(points, lidar_to_cam=lidar_to_cam[i], cam_to_img=cam_to_img[i])
            #cloud = project_image_to_cam(points, cam_to_img=cam_to_img[i])
            valid_inds = (cloud[:, 0] < 46.8) & \
                    (cloud[:, 0] >= 2) & \
                    (cloud[:, 1] < 30.8) & \
                    (cloud[:, 1] >= -30.8) & \
                    (cloud[:, 2] < 1) & \
                    (cloud[:, 2] >= -3)
            cloud = cloud[valid_inds]
            lst.append(cloud)
        return lst
        # stacked_lst = torch.stack(lst)
        # return stacked_lst
    def normalize(self, image_features):
            image_features = image_features.squeeze(0).cpu().numpy()
            min = image_features.min()
            max = image_features.max()
            image_features = (image_features-min)/(max-min)
            image_features = (image_features*255)
            return image_features

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict

class DepthFFN_ATSS(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )
        if model_cfg.get('CHANNEL_REDUCE',None) is not None:
            self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        else:
            self.channel_reduce = None
        self.sem_neck = build_neck(model_cfg.SEM_NECK)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return None
        # return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]

        offset = ddn_result["offset"]

        # #Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)
        
        batch_dict['sem_features'] = self.sem_neck([image_features])
        # # Create image feature plane-sweep volume
        # frustum_features = self.create_frustum_features(image_features=image_features,
        #                                                 depth_logits=depth_logits)
        frustum_features = self.create_frustum_features(image_features=batch_dict['sem_features'][0],
                                                        depth_logits=depth_logits)
        batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits

            self.forward_ret_dict["offset"] = offset
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict



class DepthFFN_W(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        # Create modules
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=model_cfg.DDN.BACKBONE_NAME,
            **model_cfg.DDN.ARGS
        )
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](
            disc_cfg=self.disc_cfg,
            downsample_factor=downsample_factor,
            **model_cfg.LOSS.ARGS
        )
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]
        offset = ddn_result["offset"]
        # Channel reduce
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        # Create image feature plane-sweep volume
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)
        batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits
            self.forward_ret_dict["offset"] = offset
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict
