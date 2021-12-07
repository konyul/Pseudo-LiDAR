from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
#from torch_scatter import scatter_mean

from itertools import repeat

class PointToBEV(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
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
        self.X = self.model_cfg.X
        self.Y = self.model_cfg.Y

    
    # def gen_feature_diffused_tensor(self, pc_rect, feature_z, feature_x, grid_3D_extended,
    #                                 diffused=False):
    #     valid_inds = (pc_rect[:, 2] < 70) & \
    #                 (pc_rect[:, 2] >= 0) & \
    #                 (pc_rect[:, 0] < 40) & \
    #                 (pc_rect[:, 0] >= -40) & \
    #                 (pc_rect[:, 1] < 2.5) & \
    #                 (pc_rect[:, 1] >= -1)
    #     pc_rect = pc_rect[valid_inds]
    #     # import pickle
    #     # with open('../../pcd_list/e2e_data.pickle','wb') as f:
    #     #     pickle.dump(pc_rect,f)

    #     pc_rect_quantized = torch.floor(
    #         pc_rect[:, :3] / 0.1).long().detach()
    #     pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
    #         + feature_x / 2
    #     pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + 10


    #     pc_rect_quantized += 1
    #     pc_rect_quantized = pc_rect_quantized.cuda()

    #     pc_rect_quantized_unique, inverse_idx = torch.unique(pc_rect_quantized, dim=0, return_inverse=True)

    #     data = ((
    #         grid_3D_extended[
    #             pc_rect_quantized[:, 1],
    #             pc_rect_quantized[:, 2],
    #             pc_rect_quantized[:, 0]] - pc_rect) ** 2)

    #     pc_rect_assign = torch.exp(-((
    #         grid_3D_extended[
    #             pc_rect_quantized[:, 1],
    #             pc_rect_quantized[:, 2],
    #             pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01)

    #     pc_rect_assign_unique = scatter_mean(pc_rect_assign, inverse_idx)

    #     BEV_feature = torch.zeros(
    #         (35+2, feature_z+2, feature_x+2), dtype=torch.float).cuda()
    #     BEV_feature[pc_rect_quantized_unique[:, 1],
    #                 pc_rect_quantized_unique[:, 2],
    #                 pc_rect_quantized_unique[:, 0]] = pc_rect_assign_unique

    #     if diffused:
    #         for dx in range(-1, 2):
    #             for dy in range(-1, 2):
    #                 for dz in range(-1, 2):
    #                     if dx == 0 and dy == 0 and dz == 0:
    #                         continue
    #                     else:
    #                         pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] + dx
    #                         pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + dy
    #                         pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] + dz

    #                         pc_rect_quantized_unique, inverse_idx = torch.unique(
    #                             pc_rect_quantized, dim=0, return_inverse=True)

    #                         pc_rect_assign = torch.exp(-((
    #                             grid_3D_extended[
    #                                 pc_rect_quantized[:, 1],
    #                                 pc_rect_quantized[:, 2],
    #                                 pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01) / 26

    #                         pc_rect_assign_unique = scatter_mean(
    #                             pc_rect_assign, inverse_idx)

    #                         BEV_feature[pc_rect_quantized_unique[:, 1],
    #                                     pc_rect_quantized_unique[:, 2],
    #                                     pc_rect_quantized_unique[:, 0]] = \
    #                             BEV_feature[pc_rect_quantized_unique[:, 1],
    #                                         pc_rect_quantized_unique[:, 2],
    #                                         pc_rect_quantized_unique[:, 0]] + \
    #                                 pc_rect_assign_unique

    #                         pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] - dx
    #                         pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] - dy
    #                         pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] - dz


    #     return BEV_feature[1:-1, 1:-1, 1:-1]

    # def get_3D_global_grid_extended(self, zsize, xsize, ysize):
    #     z = torch.linspace(0.0 - 70.0 / zsize / 2, 70.0 +
    #                    70.0 / zsize / 2, steps=zsize+2)
    #     x = torch.linspace(-40.0 - 80.0 / xsize / 2, 40.0 +
    #                     80.0 / xsize / 2, steps=xsize+2)
    #     y = torch.linspace(-1.0 - 3.5 / ysize / 2, 2.5 +
    #                     3.5 / ysize / 2, steps=ysize+2)
    #     pc_grid = torch.zeros((ysize+2, zsize+2, xsize+2, 3), dtype=torch.float)
    #     pc_grid[:, :, :, 0] = x.reshape(1, 1, -1)
    #     pc_grid[:, :, :, 1] = y.reshape(-1, 1, 1)
    #     pc_grid[:, :, :, 2] = z.reshape(1, -1, 1)
    #     return pc_grid

    def gen_feature_diffused_tensor(self, pc_rect, feature_x, feature_y, grid_3D_extended, batch_dict,
                                    diffused=False):
        valid_inds = (pc_rect[:, 0] < 70) & \
                    (pc_rect[:, 0] >= 0) & \
                    (pc_rect[:, 1] < 40) & \
                    (pc_rect[:, 1] >= -40) & \
                    (pc_rect[:, 2] < 1.0) & \
                    (pc_rect[:, 2] >= -3)
        pc_rect = pc_rect[valid_inds]
        # import pickle
        # with open('../pcd_list/e2e_data.pickle','wb') as f:
        #     pickle.dump(pc_rect,f)

        pc_rect_quantized = torch.floor(
            pc_rect[:, :3] / 0.1).long().detach()
        pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] \
            + feature_y / 2
        # pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] + 10
        pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] + 30


        pc_rect_quantized += 1
        pc_rect_quantized = pc_rect_quantized.cuda()

        pc_rect_quantized_unique, inverse_idx = torch.unique(pc_rect_quantized, dim=0, return_inverse=True)
        #data = ((grid_3D_extended[pc_rect_quantized[:, 1],pc_rect_quantized[:, 2],pc_rect_quantized[:, 0]] - pc_rect) ** 2)

        pc_rect_assign = torch.exp(-((
            grid_3D_extended[
                pc_rect_quantized[:, 2],
                pc_rect_quantized[:, 1],
                pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01)

        pc_rect_assign_unique = scatter_mean(pc_rect_assign, inverse_idx)

        # BEV_feature = torch.zeros(
        #     (35+2, feature_y+2, feature_x+2), dtype=torch.float).cuda()
        BEV_feature = torch.zeros(
            (self.num_bev_features+2, feature_y+2, feature_x+2), dtype=torch.float).cuda()
    
        BEV_feature[pc_rect_quantized_unique[:, 2],
                    pc_rect_quantized_unique[:, 1],
                    pc_rect_quantized_unique[:, 0]] = pc_rect_assign_unique
    

        if diffused:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        else:
                            pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] + dx
                            pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + dy
                            pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] + dz

                            pc_rect_quantized_unique, inverse_idx = torch.unique(
                                pc_rect_quantized, dim=0, return_inverse=True)

                            pc_rect_assign = torch.exp(-((
                                grid_3D_extended[
                                    pc_rect_quantized[:, 2],
                                    pc_rect_quantized[:, 1],
                                    pc_rect_quantized[:, 0]] - pc_rect) ** 2).sum(dim=1) / 0.01) / 26

                            pc_rect_assign_unique = scatter_mean(
                                pc_rect_assign, inverse_idx)

                            BEV_feature[pc_rect_quantized_unique[:, 2],
                                        pc_rect_quantized_unique[:, 1],
                                        pc_rect_quantized_unique[:, 0]] = \
                                BEV_feature[pc_rect_quantized_unique[:, 2],
                                            pc_rect_quantized_unique[:, 1],
                                            pc_rect_quantized_unique[:, 0]] + \
                                    pc_rect_assign_unique

                            pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] - dx
                            pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] - dy
                            pc_rect_quantized[:, 2] = pc_rect_quantized[:, 2] - dz


        # return BEV_feature[1:-1, 1:-1, 1:-1]
        return BEV_feature[1:-1, 101:-101, 101:-101]
    # def get_3D_global_grid_extended(self, xsize, ysize, zsize):
    #     x = torch.linspace(0.0 - 70.0 / xsize / 2, 70.0 +
    #                    70.0 / xsize / 2, steps=xsize+2)
    #     y = torch.linspace(-40.0 - 80.0 / ysize / 2, 40.0 +
    #                     80.0 / ysize / 2, steps=ysize+2)
    #     z = torch.linspace(-1.0 - 3.5 / zsize / 2, 2.5 +
    #                     3.5 / zsize / 2, steps=zsize+2)
    #     pc_grid = torch.zeros((zsize+2, ysize+2, xsize+2, 3), dtype=torch.float)
    #     pc_grid[:, :, :, 0] = x.reshape(1, 1, -1)
    #     pc_grid[:, :, :, 1] = y.reshape(1, -1, 1)
    #     pc_grid[:, :, :, 2] = z.reshape(-1, 1, 1)
    #     return pc_grid
    def get_3D_global_grid_extended(self, xsize, ysize, zsize):
        x = torch.linspace(0.0 - 70.0 / xsize / 2, 70.0 +
                       70.0 / xsize / 2, steps=xsize+2)
        y = torch.linspace(-40.0 - 80.0 / ysize / 2, 40.0 +
                        80.0 / ysize / 2, steps=ysize+2)
        z = torch.linspace(-3.0 - 4.0 / zsize / 2, 1.0 +
                        4.0 / zsize / 2, steps=zsize+2)
        pc_grid = torch.zeros((zsize+2, ysize+2, xsize+2, 3), dtype=torch.float)
        pc_grid[:, :, :, 0] = x.reshape(1, 1, -1)
        pc_grid[:, :, :, 1] = y.reshape(1, -1, 1)
        pc_grid[:, :, :, 2] = z.reshape(-1, 1, 1)
        return pc_grid


    def forward(self, batch_dict):
        grid_3D_extended = self.get_3D_global_grid_extended(self.X, self.Y, self.num_bev_features).cuda().float()   
        inputs = []
        for i in range(batch_dict['batch_size']):
            ptc = batch_dict["points"][i][:,:3].clone()
            voxel = self.gen_feature_diffused_tensor(
                            ptc, self.X, self.Y, grid_3D_extended, batch_dict, diffused=True)
            inputs.append(voxel)
        inputs = torch.stack(inputs)
        batch_dict["depth_bev_features"] = inputs
        return batch_dict


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0
    
def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        if index.numel() > 0:
            index = index.view(index_size).expand_as(src)
        else:  # pragma: no cover
            # PyTorch has a bug when view is used on zero-element tensors.
            index = src.new_empty(index_size, dtype=torch.long)

    # Broadcasting capabilties: Expand dimensions to match.
    if src.dim() != index.dim():
        raise ValueError(
            ('Number of dimensions of src and index tensor do not match, '
             'got {} and {}').format(src.dim(), index.dim()))

    expand_size = []
    for s, i in zip(src.size(), index.size()):
        expand_size += [-1 if s == i and s != 1 and i != 1 else max(i, s)]
    src = src.expand(expand_size)
    index = index.expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Sums all values from the :attr:`src` tensor into :attr:`out` at the indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`. For
    each value in :attr:`src`, its output index is specified by its index in
    :attr:`input` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`. If
    multiple indices reference the same location, their **contributions add**.

    Formally, if :attr:`src` and :attr:`index` are n-dimensional tensors with
    size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})` and
    :attr:`dim` = `i`, then :attr:`out` must be an n-dimensional tensor with
    size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`. Moreover, the
    values of :attr:`index` must be between `0` and `out.size(dim) - 1`.
    Both :attr:`src` and :attr:`index` are broadcasted in case their dimensions
    do not match.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_add

        src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_add(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0., 0., 4., 3., 3., 0.],
               [2., 4., 4., 0., 0., 0.]])
    """
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/mean.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Averages all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.If multiple indices reference the same location, their
    **contributions average** (`cf.` :meth:`~torch_scatter.scatter_add`).

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \frac{1}{N_i} \cdot
        \sum_j \mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`. :math:`N_i` indicates the number of indices
    referencing :math:`i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. (default: :obj:`None`)
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor is
            returned. (default: :obj:`None`)
        fill_value (int, optional): If :attr:`out` is not given, automatically
            fill output tensor with :attr:`fill_value`. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_scatter import scatter_mean

        src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
        index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
        out = src.new_zeros((2, 6))

        out = scatter_mean(src, index, out=out)

        print(out)

    .. testoutput::

       tensor([[0.0000, 0.0000, 4.0000, 3.0000, 1.5000, 0.0000],
               [1.0000, 4.0000, 2.0000, 0.0000, 0.0000, 0.0000]])
    """
    out = scatter_add(src, index, dim, out, dim_size, fill_value)
    count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
    return out / count.clamp(min=1)
