import torch
import torch.nn as nn
from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v, p2f
from ...backbones_3d import vfe
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

class ImageVFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, depth_downsample_factor, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.downsample_factor = depth_downsample_factor
        self.module_topology = [
            'ffn', 'f2v', 'p2f'
        ]
        self.build_modules()

    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_ffn(self):
        """
        Builds frustum feature network
        Returns:
            ffn_module: nn.Module, Frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

    def build_f2v(self):
        """
        Builds frustum to voxel transformation
        Returns:
            f2v_module: nn.Module, Frustum to voxel transformation
        """
        if self.model_cfg.get('F2V', None) is None:
            return None
        f2v_module = f2v.__all__[self.model_cfg.F2V.NAME](
            model_cfg=self.model_cfg.F2V,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg
        )
        return f2v_module

    def build_p2f(self):
        """
        Builds pseudolidar to feature transformation
        Returns:
            p2f_module: nn.Module, pseudolidar to feature transformation
        """
        if self.model_cfg.get('P2F', None) is None:
            return None
        p2f_module = p2f.__all__[self.model_cfg.P2F.NAME](
            model_cfg=self.model_cfg.P2F,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg
        )
        return p2f_module

    def get_output_feature_dim(self):
        """
        Gets number of output channels
        Returns:
            out_feature_dim: int, Number of output channels
        """
        out_feature_dim = self.ffn.get_output_feature_dim()
        return out_feature_dim

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
            **kwargs:
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        batch_dict = self.ffn(batch_dict)
        if self.model_cfg.get('F2V', None) is not None:
            batch_dict = self.f2v(batch_dict)
        if self.model_cfg.get('P2F', None) is not None:
            batch_dict = self.p2f(batch_dict)
        return batch_dict

    def get_loss(self):
        """
        Gets DDN loss
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """

        loss, tb_dict = self.ffn.get_loss()
        return loss, tb_dict
