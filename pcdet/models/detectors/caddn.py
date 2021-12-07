from .detector3d_template import Detector3DTemplate
from .. import dense_heads

class CaDDN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.vfe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }
        loss = loss_rpn + loss_depth
        return loss, tb_dict, disp_dict


class CaDDN_ATSS(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head_2d', 'dense_head',  'point_head', 'roi_head'
        ]
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.vfe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth

        if getattr(self, 'dense_head_2d', None):
            loss_rpn_2d, tb_dict = self.dense_head_2d.get_loss(batch_dict, tb_dict)
            tb_dict['loss_rpn2d'] = loss_rpn_2d.item()
            loss += loss_rpn_2d
            
        return loss, tb_dict, disp_dict

    def build_dense_head_2d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_2D', None) is None:
            return None, model_info_dict
        if self.model_cfg.DENSE_HEAD_2D.NAME == 'MMDet2DHead':
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_2D.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD_2D
            )
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict
        else:
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD_2D.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD_2D,
                input_channels=32,
                num_class=self.num_class,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            )
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict