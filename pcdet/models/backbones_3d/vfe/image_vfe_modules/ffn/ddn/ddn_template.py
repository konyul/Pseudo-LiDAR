from collections import OrderedDict
from pathlib import Path
from torch import hub

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

    
class DDNTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None,
                depth_feat_extract_layer=None, offset=False):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        self.depth_feat_extract_layer = depth_feat_extract_layer
        self.offset = offset
        if self.offset:
            in_planes = 256
            out_planes = 81
            self.semantic1 = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                            padding=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_planes),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_planes, 81, kernel_size=3, padding=1, stride=1, bias=False))
        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # Model
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {
            feat_extract_layer: 'features',
            'layer2': 'layer2',
            'layer3': 'layer3',
            **self.model.backbone.return_layers
        }

        if self.depth_feat_extract_layer == 4:
            in_planes = 81
            out_planes = 81
            self.downsampling = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=7,
                                                padding=3, stride=2, bias=False),
                                                nn.BatchNorm2d(out_planes),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1, dilation=1, ceil_mode=False))
                                                
    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss)

        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            
            # Download pretrained model if not available yet
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict,
                                                          pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def off_regress(self, off):
        "Regress offsets in [0, 1] range"
        off = torch.tanh(off)
        off = torch.clamp(off, min=-0.5, max=0.5) + 0.5
        return off

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        features = self.model.backbone(x)
        result['features'] = features['features']
        feat_shape = features['features'].shape[-2:]

        # Prediction classification logits
        x = features["out"]
        if self.depth_feat_extract_layer is None:
            x = self.model.classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["logits"] = x

        elif self.depth_feat_extract_layer == 0:
            for idx in range(len(self.model.classifier)):
                x = self.model.classifier[idx](x)
                if self.depth_feat_extract_layer == idx:
                    x_int = F.interpolate(x.clone(), size=feat_shape, mode='bilinear', align_corners=False)
                    result['depth_features'] = x_int
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["logits"] = x

        # elif self.depth_feat_extract_layer == 4:
        #     x_int = x.clone()
        #     for idx in range(len(self.model.classifier)):
        #         if idx == 0:
        #             x_int = self.model.classifier[idx](x_int)
        #             x_int = F.interpolate(x_int.clone(), size=(375,1242), mode='bilinear', align_corners=False)
        #         else:
        #             x_int = self.model.classifier[idx](x_int)
        #     result['logits_ups'] = x_int
        #     x = self.downsampling(x_int)
        #     result['logits'] = x
        elif self.depth_feat_extract_layer == 4:
            for idx in range(len(self.model.classifier)):
                if idx == 0:
                    x = self.model.classifier[idx](x)
                    x_int = x.clone()
                    x_int = F.interpolate(x_int.clone(), size=(375,1242), mode='bilinear', align_corners=False)
                else:
                    x = self.model.classifier[idx](x)
                    x_int = self.model.classifier[idx](x_int)
            result['logits_ups'] = x_int
            # x = self.downsampling(x_int)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result['logits'] = x

        result['logits_ones'] = torch.ones(x.shape[0],self.num_classes,result['features'].shape[-2],result['features'].shape[-1],device='cuda',requires_grad=False)
        if self.offset:
            result['offset'] = self.off_regress(self.semantic1(x_int))

        # Prediction auxillary classification logits
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            mask = torch.isnan(x)

            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
