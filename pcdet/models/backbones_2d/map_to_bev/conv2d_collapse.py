import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        return batch_dict

class Conv2DCollapse_w_pillar(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        self.nx, self.ny, _ = grid_size
        self.nz = 1
        if self.model_cfg.get("PLONLY",None) is not None:
            self.pl_only = self.model_cfg.PLONLY
        else:
            self.pl_only = None
    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """


        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        if not self.pl_only:
            batch_dict['pillar_spatial_features'] = batch_spatial_features
            voxel_features = batch_dict["voxel_features"]
            bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
            bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
            batch_dict["spatial_features"] = bev_features    
        else:
            batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
