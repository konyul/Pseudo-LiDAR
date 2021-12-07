import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        if self.model_cfg.get('FUSION',None) == 'pillar':
            c_in_list = [input_channels * 2, *num_filters[:-1]]
        elif self.model_cfg.get('FUSION',None) == 'occupy':
            c_in_list = [input_channels + 40, *num_filters[:-1]]
        else:
            c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        if self.model_cfg.get('FUSION',None) == 'pillar':
            self.fusion = 'pillar'
            self.monogate = BasicGate(input_channels*2, num_conv = 3)
            self.bigate = BiGate(input_channels, input_channels)
        elif self.model_cfg.get('FUSION',None) == 'occupy':
            self.fusion = 'occupy'
            self.monogate = BasicGate(input_channels + 40, num_conv = 3)
            self.bigate = BiGate(input_channels, 40)
        else:
            self.fusion = False

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        
        if self.fusion == 'pillar':
            pillar_spatial_features = data_dict['pillar_spatial_features']

            _fused_spatial_features = torch.cat((spatial_features.clone(), pillar_spatial_features.clone()), dim = 1)

            ## monogate
            #_fused_spatial_features = self.monogate(fused_spatial_features,fused_spatial_features)

            ## bigate
            # _spatial_features, _pillar_spatial_features = self.bigate(spatial_features.clone(), pillar_spatial_features.clone())
            # _fused_spatial_features = torch.cat((_spatial_features, _pillar_spatial_features), dim = 1)

            x = _fused_spatial_features
            
        elif self.fusion == 'occupy':
            depth_spatial_features = data_dict["depth_bev_features"]
            depth_spatial_features = F.interpolate(depth_spatial_features.clone(), size=spatial_features.shape[-2:], mode='bilinear', align_corners=False)
            _fused_spatial_features = torch.cat((spatial_features, depth_spatial_features), dim = 1)

            ## monogate
            #_fused_spatial_features = self.monogate(fused_spatial_features,fused_spatial_features)

            ## bigate
            # _spatial_features, _pillar_spatial_features = self.bigate(spatial_features.clone(), depth_spatial_features.clone())
            # _fused_spatial_features = torch.cat((_spatial_features, _pillar_spatial_features), dim = 1)

            x = _fused_spatial_features
            
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class BasicGate(nn.Module):
    def __init__(self, g_channel, num_conv = 3):
        super(BasicGate, self).__init__()
        self.g_channel = g_channel
        self.spatial_basic = []
        for idx in range(num_conv - 1):
            self.spatial_basic.append(
                nn.Conv2d(self.g_channel,
                          self.g_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.spatial_basic.append(
                nn.BatchNorm2d(self.g_channel, eps=1e-3, momentum=0.01)
            )
            self.spatial_basic.append(
                nn.ReLU()
            )
    
        self.spatial_basic.append(
            nn.Conv2d(self.g_channel, 1, kernel_size=3, stride=1, padding=1))
        self.spatial_basic = nn.Sequential(*self.spatial_basic)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feature, attention):
        attention = self.spatial_basic(attention)
        attention_map = torch.sigmoid(attention)
        return feature * attention_map

class BiGate(nn.Module):
    def __init__(self, g_channel, g_channel_):
        super(BiGate, self).__init__()
        self.g_channel = g_channel
        self.g_channel_ = g_channel_
        self.b_conv2d = nn.Conv2d(self.g_channel,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.a_conv2d = nn.Conv2d(self.g_channel_,
                                  1,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
    def forward(self, feat1, feat2):
        feat1_map = self.b_conv2d(feat1)
        feat1_scale = torch.sigmoid(feat1_map)
        feat2_map = self.a_conv2d(feat2)
        feat2_scale = torch.sigmoid(feat2_map)
        return feat1 * feat2_scale, feat2 * feat1_scale

