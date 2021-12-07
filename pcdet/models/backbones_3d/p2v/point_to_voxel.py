import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from torch.nn import functional as F
from ....ops.voxel import Voxelization


class PointToVoxel(nn.Module):
    
    def __init__(self, model_cfg, point_cloud_range, voxel_size, grid_size, depth_downsample_factor):
        super().__init__()
        self.model_cfg=model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.detph_downsample_factor = depth_downsample_factor
        self.voxel_layer = Voxelization(**self.model_cfg.VOXEL_LAYER)

    def extract_feat(self, batch_dict):
        points = batch_dict["points"]
        voxels, num_points, coors = self.voxelize(points)
        batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords'] = voxels, num_points, coors
        return batch_dict
        
    
    def forward(self, batch_dict):
        x = self.extract_feat(batch_dict)
        return x
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
