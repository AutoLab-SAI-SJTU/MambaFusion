from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from mamba_ssm import Block2 as MambaBlock
from mamba_ssm import Block3 as MambaBlock2
from torch.nn import functional as F

# from ..model_utils.retnet_attn import Block as RetNetBlock
# from ..model_utils.rwkv_cls import Block as RWKVBlock
# from ..model_utils.vision_lstm2 import xLSTM_Block
# from ..model_utils.ttt import TTTBlock

from ...utils.spconv_utils import replace_feature, spconv
import torch.utils.checkpoint as cp
from pcdet.models.model_utils.dsvt_utils import get_window_coors, get_inner_win_inds_cuda, get_pooling_index, get_continous_inds
from pcdet.ops.win_coors.flattened_window_cuda import get_window_coors_shift_v2 as get_window_coors_shift_v2_cuda
from pcdet.ops.win_coors.flattened_window_cuda import flattened_window_mapping as flattened_window_mapping_cuda
from pcdet.ops.win_coors.flattened_window_cuda import get_window_coors_shift_v3 as get_window_coors_shift_v3_cuda
from pcdet.ops.win_coors.flattened_window_cuda import expand_selected_coords as expand_selected_coords_cuda
@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win
# class FlattenedWindowMappingacc(nn.Module):
#     def __init__(
#             self,
#             window_shape,
#             group_size,
#             shift,
#             win_version='v2',
#             use_expand = False,
#             use_divide_token=False,
#     ) -> None:
#         super().__init__()
#         self.window_shape = window_shape
#         self.group_size = group_size
#         self.win_version = win_version
#         self.shift = shift
#         self.use_expand = use_expand
#         self.coef = 2 if self.use_expand else 1
#         self.use_divide_token = use_divide_token

        

#     def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        

#         coords = coords.long()
#         _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
#         batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
#         num_per_batch_p = (
#                 torch.div(
#                     batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
#                     self.group_size,
#                     rounding_mode="trunc",
#                 )
#                 * self.group_size
#         )

#         batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
#         flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device) 从原始坐标到基于窗口表示的坐标
#         win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

#         for i in range(batch_size):
#             if num_per_batch[i] != num_per_batch_p[i]:
                
#                 bias_index = batch_start_indices_p[i] - batch_start_indices[i]
#                 flat2win[
#                     batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
#                     batch_start_indices_p[i + 1]
#                     ] = flat2win[
#                         batch_start_indices_p[i + 1]
#                         - 2 * self.group_size
#                         + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
#                         ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
#                         win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
#                             (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
#                         : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


#             win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
#                     batch_start_indices_p[i] - batch_start_indices[i]
#             )

#             flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
#                     batch_start_indices_p[i] - batch_start_indices[i]
#             )

#         mappings = {"flat2win": flat2win, "win2flat": win2flat}


#         batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
#                                                                                     self.window_shape, self.shift)
        
#         vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
#         vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
#             self.window_shape[2] + coors_in_win[..., 0]

#         vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
#         vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
#             self.window_shape[2] + coors_in_win[..., 0]

#         if self.use_divide_token:
#             max_win_size = self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
#             x_index, mappings["x"] = torch.sort(vx)
#             y_index, mappings["y"] = torch.sort(vy)
#             x_win_index = x_index // max_win_size
#             y_win_index = y_index // max_win_size
            
#         else:
#             _, mappings["x"] = torch.sort(vx)
#             _, mappings["y"] = torch.sort(vy)


#         return mappings

def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2



    
class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2',
            use_expand = False,
            use_divide_token=False,
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift
        self.use_expand = use_expand
        self.coef = 2 if self.use_expand else 1
        self.use_divide_token = use_divide_token
        if self.win_version == 'v4':

            self.shift_list = [[[0, 0, 0], [(self.window_shape[0] + 1) // 2, (self.window_shape[1] + 1) // 2, 0]]]
            self.window_shape = [[self.window_shape, self.window_shape]]
            self.sparse_shape_list = [[96, 264, 6]]
            self.set_info = [[self.group_size, 2]]
            self.num_shifts = [2]
            self.model_cfg = {'expand_max_voxels': 30}
        

    # def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        
    #     if self.win_version == 'v4':
    #         info = {}
    #         info[f'voxel_coors_stage0'] = coords.long().clone()
    #         info = self.window_partition(info, 0) 
    #         info = self.get_set(info, 0)
    #         inds_list = [[info[f'set_voxel_inds_stage{s}_shift{i}']
    #                     for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
    #         return inds_list[0]
    #     else:
    #         coords = coords.long()
    #         _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
    #         batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
    #         num_per_batch_p = (
    #                 torch.div(
    #                     batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
    #                     self.group_size,
    #                     rounding_mode="trunc",
    #                 )
    #                 * self.group_size
    #         )

    #         batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
    #         flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device) 从原始坐标到基于窗口表示的坐标
    #         win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

    #         for i in range(batch_size):
    #             if num_per_batch[i] != num_per_batch_p[i]:
                    
    #                 bias_index = batch_start_indices_p[i] - batch_start_indices[i]
    #                 flat2win[
    #                     batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
    #                     batch_start_indices_p[i + 1]
    #                     ] = flat2win[
    #                         batch_start_indices_p[i + 1]
    #                         - 2 * self.group_size
    #                         + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
    #                         ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
    #                         win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
    #                             (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
    #                         : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


    #             win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
    #                     batch_start_indices_p[i] - batch_start_indices[i]
    #             )

    #             flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
    #                     batch_start_indices_p[i] - batch_start_indices[i]
    #             )
    #         win2flat_new, flat2win_new = flattened_window_mapping_cuda(
    #                 num_per_batch,
    #                 num_per_batch_p,
    #                 batch_start_indices,
    #                 batch_start_indices_p,
    #                 self.group_size,
    #                 batch_size
    #             )
    #         assert torch.allclose(win2flat, win2flat_new) and torch.allclose(flat2win, flat2win_new), 'Error in flattened_window_mapping_cuda'
    #         mappings = {"flat2win": flat2win, "win2flat": win2flat}

    #         get_win = self.win_version

    #         if get_win == 'v1':
    #             for shifted in [False]:
    #                 (
    #                     n2,
    #                     m2,
    #                     n1,
    #                     m1,
    #                     x1,
    #                     y1,
    #                     x2,
    #                     y2,
    #                 ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
    #                 vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
    #                 vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
    #                 vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
    #                 vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
    #                 _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
    #                 _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

    #         elif get_win == 'v2':
    #             batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
    #                                                                                         self.window_shape, self.shift)
    #             # batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2_cuda(coords, sparse_shape,
    #             #                                                                             self.window_shape, self.shift)
                
    #             vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
    #             vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
    #                 self.window_shape[2] + coors_in_win[..., 0]

    #             vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
    #             vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
    #                 self.window_shape[2] + coors_in_win[..., 0]
    #             vx_new, vy_new = get_window_coors_shift_v3_cuda(coords, sparse_shape, self.window_shape, self.shift)
    #             # 判断是否完全一样
    #             assert torch.allclose(vx, vx_new) and torch.allclose(vy, vy_new), 'Error in get_window_coors_shift_v3_cuda'
                   
    #             if self.use_divide_token:
    #                 max_win_size = self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
    #                 x_index, mappings["x"] = torch.sort(vx)
    #                 y_index, mappings["y"] = torch.sort(vy)
    #                 x_win_index = x_index // max_win_size
    #                 y_win_index = y_index // max_win_size
                    
    #             else:
    #                 _, mappings["x"] = torch.sort(vx)
    #                 _, mappings["y"] = torch.sort(vy)

    #         elif get_win == 'v3':
    #             batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
    #                                                                                         self.window_shape)
    #             vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
    #             vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
    #                     self.window_shape[2] + coors_in_win[..., 0]
    #             vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
    #                     self.window_shape[2] + coors_in_win[..., 0]

    #             vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
    #             vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
    #                     self.window_shape[2] + coors_in_win[..., 0]
    #             vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
    #                     self.window_shape[2] + coors_in_win[..., 0]

    #             _, mappings["x_xy"] = torch.sort(vx_xy)
    #             _, mappings["y_xy"] = torch.sort(vy_xy)
    #             _, mappings["x_yx"] = torch.sort(vx_yx)
    #             _, mappings["y_yx"] = torch.sort(vy_yx)

    #         return mappings
    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        
        if self.win_version == 'v4':
            info = {}
            info[f'voxel_coors_stage0'] = coords.long().clone()
            info = self.window_partition(info, 0) 
            info = self.get_set(info, 0)
            inds_list = [[info[f'set_voxel_inds_stage{s}_shift{i}']
                        for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
            return inds_list[0]
        else:
            coords = coords.long()
            _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
            batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
            num_per_batch_p = (
                    torch.div(
                        batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                        self.group_size,
                        rounding_mode="trunc",
                    )
                    * self.group_size
            )

            batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
            # flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device) 从原始坐标到基于窗口表示的坐标
            # win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

            # for i in range(batch_size):
            #     if num_per_batch[i] != num_per_batch_p[i]:
                    
            #         bias_index = batch_start_indices_p[i] - batch_start_indices[i]
            #         flat2win[
            #             batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
            #             batch_start_indices_p[i + 1]
            #             ] = flat2win[
            #                 batch_start_indices_p[i + 1]
            #                 - 2 * self.group_size
            #                 + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
            #                 ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
            #                 win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
            #                     (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
            #                 : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


            #     win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
            #             batch_start_indices_p[i] - batch_start_indices[i]
            #     )

            #     flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
            #             batch_start_indices_p[i] - batch_start_indices[i]
            #     )
            win2flat, flat2win = flattened_window_mapping_cuda(
                    num_per_batch,
                    num_per_batch_p,
                    batch_start_indices,
                    batch_start_indices_p,
                    self.group_size,
                    batch_size
                )
            # assert torch.allclose(win2flat, win2flat_new) and torch.allclose(flat2win, flat2win_new), 'Error in flattened_window_mapping_cuda'
            mappings = {"flat2win": flat2win, "win2flat": win2flat}

            get_win = self.win_version

            if get_win == 'v1':
                for shifted in [False]:
                    (
                        n2,
                        m2,
                        n1,
                        m1,
                        x1,
                        y1,
                        x2,
                        y2,
                    ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                    vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                    vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                    vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                    vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                    _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                    _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

            elif get_win == 'v2':
                # batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                #                                                                             self.window_shape, self.shift)
                # batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2_cuda(coords, sparse_shape,
                #                                                                             self.window_shape, self.shift)
                
                # vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
                # vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                #     self.window_shape[2] + coors_in_win[..., 0]

                # vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
                # vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                #     self.window_shape[2] + coors_in_win[..., 0]
                vx, vy = get_window_coors_shift_v3_cuda(coords, sparse_shape, self.window_shape, self.shift)
                # 判断是否完全一样
                # assert torch.allclose(vx, vx_new) and torch.allclose(vy, vy_new), 'Error in get_window_coors_shift_v3_cuda'
                   
                if self.use_divide_token:
                    max_win_size = self.window_shape[0] * self.window_shape[1] * self.window_shape[2] * self.coef
                    x_index, mappings["x"] = torch.sort(vx)
                    y_index, mappings["y"] = torch.sort(vy)
                    x_win_index = x_index // max_win_size
                    y_win_index = y_index // max_win_size
                    
                else:
                    _, mappings["x"] = torch.sort(vx)
                    _, mappings["y"] = torch.sort(vy)

            elif get_win == 'v3':
                batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                            self.window_shape)
                vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
                vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                        self.window_shape[2] + coors_in_win[..., 0]
                vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                        self.window_shape[2] + coors_in_win[..., 0]

                vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
                vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                        self.window_shape[2] + coors_in_win[..., 0]
                vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                        self.window_shape[2] + coors_in_win[..., 0]

                _, mappings["x_xy"] = torch.sort(vx_xy)
                _, mappings["y_xy"] = torch.sort(vy_xy)
                _, mappings["x_yx"] = torch.sort(vx_yx)
                _, mappings["y_yx"] = torch.sort(vy_yx)

            return mappings
        
    def get_set(self, voxel_info, stage_id):
        '''
        This is one of the core operation of DSVT. 
        Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
        To make it clear and easy to follow, we do not use loop to process two shifts.
        Args:
            voxel_info (dict): 
                The dict contains the following keys
                - batch_win_inds_s{i} (Tensor[float]): Windows indexs of each voxel with shape (N), computed by 'window_partition'.
                - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner window of each voxel with shape (N, 3), computed by 'window_partition'.
                    Each row is (z, y, x). 
                - ...
        
        Returns:
            See from 'forward' function.
        '''
        batch_win_inds_shift0 = voxel_info[f'batch_win_inds_stage{stage_id}_shift0'] # 获取每个voxel所属的window的index
        coors_in_win_shift0 = voxel_info[f'coors_in_win_stage{stage_id}_shift0'] # 获取每个voxel在window中的相对坐标
        set_voxel_inds_shift0 = self.get_set_single_shift(batch_win_inds_shift0, stage_id, shift_id=0, coors_in_win=coors_in_win_shift0) # 得到按照y坐标排序后的内部体素索引和按照x坐标排序后的内部体素索引
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0  
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s0 = torch.roll(set_voxel_inds_shift0.clone(), shifts=1, dims=-1) # [2, num_of_set, 90] 将set_voxel_inds_shift0向右移动一位
        prefix_set_voxel_inds_s0[ :, :, 0] = -1 
        set_voxel_mask_s0 = (set_voxel_inds_shift0 == prefix_set_voxel_inds_s0) # 得到一个表示每个set中是否有重复体素的掩码
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0

        batch_win_inds_shift1 = voxel_info[f'batch_win_inds_stage{stage_id}_shift1'] # 得到window shift之后每个voxel所属的window的index 
        coors_in_win_shift1 = voxel_info[f'coors_in_win_stage{stage_id}_shift1'] # 得到window shift之后每个voxel在window中的相对坐标
        set_voxel_inds_shift1 = self.get_set_single_shift(batch_win_inds_shift1, stage_id, shift_id=1, coors_in_win=coors_in_win_shift1)
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1  
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s1 = torch.roll(set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s1[ :, :, 0] = -1
        set_voxel_mask_s1 = (set_voxel_inds_shift1 == prefix_set_voxel_inds_s1) # 得到一个表示每个set中是否有重复体素的掩码
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1

        return voxel_info
    
    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        '''
        voxel_order_list[list]: order respectively sort by x, y, z
        '''

        device = batch_win_inds.device

        # max number of voxel in a window
        voxel_num_set = self.set_info[stage_id][0] # 90 一个set中最多有90个voxel
        max_voxel = self.window_shape[stage_id][shift_id][0] * \
            self.window_shape[stage_id][shift_id][1] * \
            self.window_shape[stage_id][shift_id][2] # 900 一个window中最多有900个voxel

        if self.model_cfg.get('expand_max_voxels', None) is not None:
            max_voxel *= self.model_cfg.get('expand_max_voxels', None)
        contiguous_win_inds = torch.unique(
            batch_win_inds, return_inverse=True)[1] # [num_of_voxels] 每个voxel属于那个window
        voxelnum_per_win = torch.bincount(contiguous_win_inds) # torch.Size([num_of_window])
        win_num = voxelnum_per_win.shape[0] # num_of_window 有150个window

        setnum_per_win_float = voxelnum_per_win / voxel_num_set # torch.Size([num_of_window]) 一个window中有多少个set
        setnum_per_win = torch.ceil(setnum_per_win_float).long() # 向上取整 torch.Size([num_of_window]) 一个window中有多少个set

        set_num = setnum_per_win.sum().item() # 有多少个set num_of_set
        setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1] # torch.Size([num_of_window - 1]) 一个window中对应的最后一个voxel的编号

        set_win_inds = torch.full((set_num,), 0, device=device) # torch.Size([num_of_set]) 有308个set
        set_win_inds[setnum_per_win_cumsum] = 1 
        set_win_inds = torch.cumsum(set_win_inds, dim=0) # 每个set所在的window的index torch.Size([num_of_set])

        # input [0,0,0, 1, 2,2]
        roll_set_win_inds_left = torch.roll( 
            set_win_inds, -1)  # [0,0, 1, 2,2,0]
        diff = set_win_inds - roll_set_win_inds_left  # [0, 0, -1, -1, 0, 2] 
        end_pos_mask = diff != 0 # 得到一个表示每个set是否是窗口中的最后一个set的掩码
        template = torch.ones_like(set_win_inds)
        template[end_pos_mask] = (setnum_per_win - 1) * -1  # [1,1,-2, 0, 1,-1]
        set_inds_in_win = torch.cumsum(template, dim=0)  # [1,2,0, 0, 1,0]
        set_inds_in_win[end_pos_mask] = setnum_per_win  # [1,2,3, 1, 1,2]
        set_inds_in_win = set_inds_in_win - 1  # [0,1,2, 0, 0,1] 得到每个set在window中的index

        offset_idx = set_inds_in_win[:, None].repeat(
            1, voxel_num_set) * voxel_num_set # torch.Size([num_of_set, 90]) 每个set在window中的index乘以set中voxel的数量
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * \
            voxelnum_per_win[set_win_inds][:, None]
        base_select_idx = base_select_idx.double(
        ) / (setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        base_select_idx = torch.floor(base_select_idx)

        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel # torch.Size([num_of_set, 90]) 每个set在window中的index乘以set中voxel的数量加上window的index乘以window中voxel的数量

        # sort by y
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds) # torch.Size([num_of_voxels]) 每个voxel在window中的index 0-899
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds # torch.Size([num_of_voxels]) 每个voxel在window中的index乘以window中voxel的数量加上window的index乘以window中voxel的数量
        _, order1 = torch.sort(global_voxel_inds) # torch.Size([num_of_voxels]) 按照voxel的global index排序
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0] # torch.Size([num_of_voxels]) 每个voxel在window中的index乘以window中voxel的数量加上window的index乘以window中voxel的数量
        _, order2 = torch.sort(global_voxel_inds_sorty) # torch.Size([num_of_voxels]) 对于每个window内的voxels按照voxel的y坐标排序

        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1]) # torch.Size([num_of_voxels]) 对于每个window内的voxels按照voxel的y坐标排序
        inner_voxel_inds_sorty_reorder = inner_voxel_inds_sorty 
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty_reorder + \
            max_voxel * contiguous_win_inds # torch.Size([num_of_voxels]) 得到按照y坐标排序后的内部体素索引
        voxel_inds_padding_sorty = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device) # torch.Size([num_of_window * 900]) 生成一个全-1的tensor
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(
            0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device) # torch.Size([num_of_voxels]) 得到按照y坐标排序后的内部体素索引 -1的位置是空的，其他位置是按照y坐标排序后的内部体素索引

        # sort by x
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)

        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1])
        inner_voxel_inds_sortx_reorder = inner_voxel_inds_sortx
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx_reorder + \
            max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(
            0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device) # torch.Size([num_of_voxels]) 得到按照x坐标排序后的内部体素索引 -1的位置是空的，其他位置是按照x坐标排序后的内部体素索引

        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()] # torch.Size([num_of_set, 90]) 得到按照y坐标排序后的内部体素索引
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()] # torch.Size([num_of_set, 90]) 得到按照x坐标排序后的内部体素索引
        all_set_voxel_inds = torch.stack(
            (set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0) # torch.Size([2, num_of_set, 90]) 得到按照y坐标排序后的内部体素索引和按照x坐标排序后的内部体素索引

        return all_set_voxel_inds
    @torch.no_grad()
    def window_partition(self, voxel_info, stage_id):
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'voxel_coors_stage{stage_id}'], 
                                                        self.sparse_shape_list[stage_id], self.window_shape[stage_id][i], i == 1, self.shift_list[stage_id][i])
                                            
            voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds # 得到每个点所在的window的index
            voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win # 得到每个点在window内的坐标
        
        return voxel_info

class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2, return_abs_coords=False):
        super().__init__()
        self.dim = dim
        self.return_abs_coords = return_abs_coords

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        if out_dim == -1:
            self.norm = norm_layer(dim)
        else:
            self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

        self.num_points = 6 #3
    # def forward(self, x, coords_shift=1, diffusion_scale=4, ori_coords_height=None):
    #     assert diffusion_scale == 4 or diffusion_scale == 2
    #     # 将 spconv 操作保持原状
    #     x = self.sub_conv(x)

    #     # 可以将下面的部分用 torch.jit.script 进行加速
    #     return self._jit_forward(x, coords_shift, diffusion_scale, ori_coords_height)
    
    # @torch.jit.script
    # def _jit_forward(self, x, coords_shift=1, diffusion_scale=4, ori_coords_height=None):
    #     assert diffusion_scale==4 or diffusion_scale==2
    #     d, h, w = x.spatial_shape
    #     down_scale = self.down_scale

    #     if self.diffusion:
    #         x_feat_att = x.features.mean(-1)
    #         batch_size = x.batch_size
    #         selected_diffusion_feats_list = [x.features.clone()]
    #         selected_diffusion_coords_list = [x.indices.clone()]
    #         if self.return_abs_coords:
    #             selected_ori_coords_height_list = [ori_coords_height.clone()]
    #         for i in range(batch_size):
    #             mask = x.indices[:, 0] == i
    #             valid_num = mask.sum()
    #             K = int(valid_num * self.diff_scale)
    #             _, indices = torch.topk(x_feat_att[mask], K) # 选择最大的

    #             selected_coords_copy = x.indices[mask][indices].clone() # [3372, 4]
    #             selected_coords_num = selected_coords_copy.shape[0]
    #             selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1) # [13488, 4]
    #             feats_expand_N, feats_expand_M = x.features[mask][indices].shape
    #             selected_feats_expand = torch.zeros(feats_expand_N * diffusion_scale, feats_expand_M, device=x.features.device) # [13488, 128]
    #             # selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0 # [13488, 128]
    #             if self.return_abs_coords:
    #                 selected_ori_coords_height = ori_coords_height[mask][indices].repeat(diffusion_scale)


    #             selected_coords_expand, selected_feats_expand = expand_selected_coords_cuda(
    #                 selected_coords_copy,
    #                 diffusion_scale,
    #                 coords_shift,
    #                 h,
    #                 w,
    #                 d,
    #                 feats_expand_M  # 特征维度
    #             )
    #             selected_diffusion_coords_list.append(selected_coords_expand)
    #             selected_diffusion_feats_list.append(selected_feats_expand)
    #             # assert torch.allclose(selected_coords_expand, selected_coords_expand_new) and torch.allclose(selected_feats_expand, selected_feats_expand_new), 'Error in expand_selected_coords_cuda'
    #             if self.return_abs_coords:
    #                 selected_ori_coords_height_list.append(selected_ori_coords_height)

    #         coords = torch.cat(selected_diffusion_coords_list)
    #         final_diffusion_feats = torch.cat(selected_diffusion_feats_list)
    #         if self.return_abs_coords:
    #             final_ori_coords_height = torch.cat(selected_ori_coords_height_list)

    #     else:
    #         coords = x.indices.clone()
    #         final_diffusion_feats = x.features.clone()

    #     # if self.return_abs_coords:
    #     #     new_coords_height = new_coords_height[ma]
    #     coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
    #     coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
    #     coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

    #     scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
    #             x.spatial_shape[2] // down_scale[0])
    #     scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
    #     scale_z = (x.spatial_shape[0] // down_scale[2])


    #     merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

    #     features_expand = final_diffusion_feats

    #     new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
    #     unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

    #     x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)
    #     if self.return_abs_coords:
    #         final_ori_coords_height_merged = torch_scatter.scatter_mean(final_ori_coords_height, unq_inv, dim=0)

    #     unq_coords = unq_coords.int()
    #     voxel_coords = torch.stack((unq_coords // scale_xyz,
    #                                 (unq_coords % scale_xyz) // scale_yz,
    #                                 (unq_coords % scale_yz) // scale_z,
    #                                 unq_coords % scale_z), dim=1)
    #     voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

    #     x_merge = self.norm(x_merge)

    #     x_merge = spconv.SparseConvTensor(
    #         features=x_merge,
    #         indices=voxel_coords.int(),
    #         spatial_shape=new_sparse_shape,
    #         batch_size=x.batch_size
    #     )
    #     if self.return_abs_coords:
    #         return x_merge, unq_inv, final_ori_coords_height_merged
    #     return x_merge, unq_inv

    # @torch.jit.script
    def forward(self, x, coords_shift=1, diffusion_scale=4, ori_coords_height=None):
        assert diffusion_scale==4 or diffusion_scale==2
        x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale
        if ori_coords_height is not None and not self.diffusion:
            final_ori_coords_height = ori_coords_height.clone()
        if self.diffusion:
            x_feat_att = x.features.mean(-1)
            batch_size = x.batch_size
            selected_diffusion_feats_list = [x.features.clone()]
            selected_diffusion_coords_list = [x.indices.clone()]
            if self.return_abs_coords:
                selected_ori_coords_height_list = [ori_coords_height.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K) # 选择最大的

                selected_coords_copy = x.indices[mask][indices].clone() # [3372, 4]
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1) # [13488, 4]
                feats_expand_N, feats_expand_M = x.features[mask][indices].shape
                selected_feats_expand = torch.zeros(feats_expand_N * diffusion_scale, feats_expand_M, device=x.features.device) # [13488, 128]
                # selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0 # [13488, 128]
                if self.return_abs_coords:
                    selected_ori_coords_height = ori_coords_height[mask][indices].repeat(diffusion_scale)


#                 selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (
#                             selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
#                 selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (
#                             selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
#                 selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (
#                         selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

#                 selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (
#                         selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
#                 selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (
#                         selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
#                 selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (
#                     selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

#                 if diffusion_scale==4:
# #                         print('####diffusion_scale==4')
#                     selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (
#                         selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
#                     selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (
#                         selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
#                     selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (
#                     selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

#                     selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (
#                             selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
#                     selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (
#                             selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
#                     selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (
#                         selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

#                 selected_diffusion_coords_list.append(selected_coords_expand)
#                 selected_diffusion_feats_list.append(selected_feats_expand)
                selected_coords_expand, selected_feats_expand = expand_selected_coords_cuda(
                    selected_coords_copy,
                    diffusion_scale,
                    coords_shift,
                    h,
                    w,
                    d,
                    feats_expand_M  # 特征维度
                )
                if self.return_abs_coords:
                    ori_coords = x.indices[mask]
                    ori_coords_flatten = (ori_coords[:, 1] // 2) * w * h + ori_coords[:, 2] * w + ori_coords[:, 3]
                    selected_coords_flatten = (selected_coords_expand[:, 1] // 2) * w * h + selected_coords_expand[:, 2] * w + selected_coords_expand[:, 3]
                    sorted_coords, sorted_indices = torch.sort(selected_coords_flatten)
                    shifted_sorted_coords = torch.cat((torch.tensor([True], device=selected_coords_flatten.device), sorted_coords[1:] != sorted_coords[:-1]))
                    unique_mask = torch.zeros_like(shifted_sorted_coords, dtype=torch.bool)
                    unique_mask[sorted_indices] = shifted_sorted_coords
                    
                    mask_new = ~torch.isin(selected_coords_flatten, ori_coords_flatten)
                    mask_new = mask_new & unique_mask
                    selected_diffusion_coords_list.append(selected_coords_expand[mask_new])
                    selected_diffusion_feats_list.append(selected_feats_expand[mask_new])
                    selected_ori_coords_height_list.append(selected_ori_coords_height[mask_new])
                    # selected_diffusion_coords_list.append(selected_coords_expand)
                    # selected_diffusion_feats_list.append(selected_feats_expand)
                    # selected_ori_coords_height_list.append(selected_ori_coords_height)
                else:
                    selected_diffusion_coords_list.append(selected_coords_expand)
                    selected_diffusion_feats_list.append(selected_feats_expand)
                # assert torch.allclose(selected_coords_expand, selected_coords_expand_new) and torch.allclose(selected_feats_expand, selected_feats_expand_new), 'Error in expand_selected_coords_cuda'
                # if self.return_abs_coords:
                #     selected_ori_coords_height_list.append(selected_ori_coords_height)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)
            if self.return_abs_coords:
                final_ori_coords_height = torch.cat(selected_ori_coords_height_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        # if self.return_abs_coords:
        #     new_coords_height = new_coords_height[ma]
        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (x.spatial_shape[0] // down_scale[2]) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (x.spatial_shape[0] // down_scale[2])


        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)
        if self.return_abs_coords:
            # assert torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)[2].max() == 2
            final_ori_coords_height_merged = torch_scatter.scatter_mean(final_ori_coords_height, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        x_merge = self.norm(x_merge)

        x_merge = spconv.SparseConvTensor(
            features=x_merge,
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        if self.return_abs_coords:
            return x_merge, unq_inv, final_ori_coords_height_merged
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c))
        up_x = up_x.replace_feature(up_x.features + x_copy)
        return up_x


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'Mamba2': MambaBlock2,
    # 'RWKV': RWKVBlock,
    # 'RetNet': RetNetBlock,
    # 'xLSTM': xLSTM_Block,
    # 'TTT': TTTBlock,
}


class LIONLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0, win_version='v2', use_expand=False):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction
        self.win_version = win_version
        self.use_expand = use_expand

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0 ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift, self.win_version, use_expand=self.use_expand)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            if self.win_version == 'v4':
                voxel_inds = mappings[i][0]
                x_features = x.features[voxel_inds]

                x_features = block(x_features)
                flatten_inds = voxel_inds.reshape(-1) # torch.Size([num_of_window * 90])
                unique_flatten_inds, inverse = torch.unique( # 找出唯一的体素索引，并返回一个数组，该数组可以用于恢复原始的体素索引。
                    flatten_inds, return_inverse=True)
                perm = torch.arange(inverse.size( 
                    0), dtype=inverse.dtype, device=inverse.device) # torch.Size([num_of_window * 90]) 
                inverse, perm = inverse.flip([0]), perm.flip([0]) # 进行翻转
                perm = inverse.new_empty( # torch.Size([num_of_voxel + num_of_patch])
                    unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
                x_features = x_features.reshape(-1, 128)[perm] # torch.Size([num_of_window * 90, 128]) 将src2中的元素按照voxel_inds的唯一值的顺序进行排序。
                x = replace_feature(x, x_features)
            else: 
                indices = mappings[self.direction[i]]
                x_features = x.features[indices][mappings["flat2win"]]
                x_features = x_features.view(-1, self.group_size, x.features.shape[-1])

                x_features = block(x_features)

                x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x

class LocalMambaLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0, win_version='v2', use_expand=False, use_checkpoint=True, use_inverse=False):
        super(LocalMambaLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction
        self.win_version = win_version
        self.use_expand = use_expand
        self.use_inverse = use_inverse

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0 and use_checkpoint ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            if operator_cfg['with_cp']:
                print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift, self.win_version, use_expand=self.use_expand)

    def forward(self, x, fixed_mapping=None):
        if fixed_mapping is not None:
            mappings = fixed_mapping
        else:
            mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            if self.win_version == 'v4':
                voxel_inds = mappings[i][0]
                x_features = x.features[voxel_inds]

                x_features = block(x_features)
                flatten_inds = voxel_inds.reshape(-1) # torch.Size([num_of_window * 90])
                unique_flatten_inds, inverse = torch.unique( # 找出唯一的体素索引，并返回一个数组，该数组可以用于恢复原始的体素索引。
                    flatten_inds, return_inverse=True)
                perm = torch.arange(inverse.size( 
                    0), dtype=inverse.dtype, device=inverse.device) # torch.Size([num_of_window * 90]) 
                inverse, perm = inverse.flip([0]), perm.flip([0]) # 进行翻转
                perm = inverse.new_empty( # torch.Size([num_of_voxel + num_of_patch])
                    unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
                x_features = x_features.reshape(-1, 128)[perm] # torch.Size([num_of_window * 90, 128]) 将src2中的元素按照voxel_inds的唯一值的顺序进行排序。
                x = replace_feature(x, x_features)
            else: 
                indices = mappings[self.direction[i]]
                x_features = x.features[indices][mappings["flat2win"]]
                x_features = x_features.view(-1, self.group_size, x.features.shape[-1])
                if self.use_inverse:
                    x_features = torch.flip(x_features, dims=[1])

                x_features = block(x_features)

                if self.use_inverse:
                    x_features = torch.flip(x_features, dims=[1])
                x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x, mappings

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class LIONBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0):
        super().__init__()

        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2, n_layer))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LIONLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + 2 * (idx + depth), n_layer))
            self.decoder_norm.append(norm_fn(dim))
            
            self.upsample_list.append(PatchExpanding3D(dim))
            

    def forward(self, x):
        features = []
        index = []

        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])

            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = enc(x)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            x = dec(x)
            x = self.upsample_list[i](x, up_x, unq_inv)
            x = replace_feature(x, norm(x.features))
            i = i + 1
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
    
class LocalMamba(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0, use_offset=False, win_version='v2', use_acc=False, use_expand=False, use_fixed_mapping=False, use_inverse=False, use_checkpoint=True):
        super().__init__()

        self.down_scales = down_scales

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()
        self.use_fixed_mapping = use_fixed_mapping
        self.use_inverse = use_inverse

        norm_fn = partial(nn.LayerNorm)

        shift = [False, shift]
        for idx in range(depth):
            self.encoder.append(LocalMambaLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + idx * 2, n_layer, win_version=win_version, use_expand=use_expand, use_checkpoint=use_checkpoint))
            self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=window_shape[-1] + 1, num_pos_feats=dim))
            if self.use_fixed_mapping:
                pass
            else:
                self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(depth):
            self.decoder.append(LocalMambaLayer(dim, 1, window_shape, group_size, direction, shift[idx], operator, layer_id + 2 * (idx + depth), n_layer, win_version=win_version, use_expand=use_expand, use_checkpoint=use_checkpoint, use_inverse=use_inverse))
            self.decoder_norm.append(norm_fn(dim))
            if self.use_fixed_mapping:
                pass
            else:
                self.upsample_list.append(PatchExpanding3D(dim))
            

    def forward(self, x, ori_fixed_mapping_list=None):
        features = []
        index = []

        fixed_mapping_list = []
        for idx, enc in enumerate(self.encoder):
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                         embed_layer=self.pos_emb_list[idx])

            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            if self.use_fixed_mapping and ori_fixed_mapping_list is not None:
                fixed_mapping = ori_fixed_mapping_list[idx]
            else:
                fixed_mapping = None
            x, mapping = enc(x, fixed_mapping)
            features.append(x)
            if self.use_fixed_mapping:
                fixed_mapping_list.append(mapping)
                index.append(None)
            else:
                x, unq_inv = self.downsample_list[idx](x)
                index.append(unq_inv)

        i = 0
        for dec, norm, up_x, unq_inv, up_scale in zip(self.decoder, self.decoder_norm, features[::-1],
                                                      index[::-1], self.down_scales[::-1]):
            if self.use_fixed_mapping:
                fixed_mapping = fixed_mapping_list[len(fixed_mapping_list) - i - 1]
                # if self.use_inverse:
                #     fixed_mapping['x'] = torch.flip(fixed_mapping['x'], dims=[0])
                #     fixed_mapping['y'] = torch.flip(fixed_mapping['y'], dims=[0])
                x, _ = dec(x, fixed_mapping)
                x = replace_feature(x, norm(x.features + up_x.features))
            else:
                fixed_mapping = None
                x, _ = dec(x, fixed_mapping)
                x = self.upsample_list[i](x, up_x, unq_inv)
                x = replace_feature(x, norm(x.features))
                i = i + 1
        if self.use_fixed_mapping:
            return x, fixed_mapping_list
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed   

class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats

class ParamModule(nn.Module):
    def __init__(self, lidar_param=0.5, camera_param=0.5, scale_lidar=-1.0, scale_camera=-1.0, coef=0.5):
        super(ParamModule, self).__init__()
        # 定义一些可学习参数
        self.lidar_param = nn.Parameter(torch.tensor(lidar_param), requires_grad=True)
        self.camera_param = nn.Parameter(torch.tensor(camera_param), requires_grad=True)
        self.scale_lidar = nn.Parameter(torch.tensor(scale_lidar), requires_grad=True)
        self.scale_camera = nn.Parameter(torch.tensor(scale_camera), requires_grad=True)
        self.coef = coef

    def forward(self):
        # 在forward中返回参数，或者你可以在这里处理这些参数
        scale_lidar_pos = torch.sigmoid(self.scale_lidar) * self.coef
        scale_camera_pos = torch.sigmoid(self.scale_camera) * self.coef
        return self.lidar_param, self.camera_param, scale_lidar_pos, scale_camera_pos
    
#for waymo and nuscenes, kitti, once
class LION3DBackboneOneStride(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR

        # 新增
        self.use_prebackbone = model_cfg.get('USE_PREBACKBONE', False)
        self.use_height_fidelity = model_cfg.get('RETURN_ABS_COORDS', False)
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer)  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, return_abs_coords=self.use_height_fidelity)
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer)

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, return_abs_coords=self.use_height_fidelity)


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer)

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, return_abs_coords=self.use_height_fidelity)

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer)

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale, return_abs_coords=self.use_height_fidelity)

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)
        self.use_dow5 = model_cfg.get('USE_DOW5', False)
        self.dow5_diff = model_cfg.get('DOW5_DIFF', True)
        if self.use_dow5:
            self.dow5 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                        norm_layer=norm_fn, diffusion=self.dow5_diff and diffusion, diff_scale=diff_scale, return_abs_coords=self.use_height_fidelity)

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape, # [32, 360, 360]
            batch_size=batch_size
        )
        if self.use_height_fidelity:
            if 'ori_coords_height' in batch_dict:
                ori_coords_height = batch_dict['ori_coords_height']
            else:
                ori_coords_height = x.indices[:, 1].to(torch.float32)
            #ori_coords_height = batch_dict['ori_coords_height']
            x = self.linear_1(x)
            x1, _, ori_coords_height = self.dow1(x, ori_coords_height=ori_coords_height)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
            batch_dict['ori_coords_height_coords1'] = ori_coords_height

            x = self.linear_2(x1)
            x2, _, ori_coords_height = self.dow2(x, ori_coords_height=ori_coords_height)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
            batch_dict['ori_coords_height_coords2'] = ori_coords_height

            x = self.linear_3(x2)
            x3, _, ori_coords_height= self.dow3(x, ori_coords_height=ori_coords_height)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
            batch_dict['ori_coords_height_coords3'] = ori_coords_height
            
            # 应该在这里插入需要融合的token
            # 可以先试试插入重点位置的token
            x = self.linear_4(x3)

            x4, _, ori_coords_height = self.dow4(x, ori_coords_height=ori_coords_height)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
            batch_dict['ori_coords_height_coords4'] = ori_coords_height
            x = self.linear_out(x4)
            
            if self.use_dow5:
                x, _, ori_coords_height = self.dow5(x, ori_coords_height=ori_coords_height)  ## 18.5k --> 18.5k  [2, 1000, 1000]-->[1, 1000, 1000]
            batch_dict['ori_coords_height'] = ori_coords_height
        else:
            x = self.linear_1(x)
            x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]
            x = self.linear_2(x1)
            x2, _ = self.dow2(x)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]
            x = self.linear_3(x2)
            x3, _ = self.dow3(x)   ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]
            x = self.linear_4(x3)

            x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
            x = self.linear_out(x4)
            
            if self.use_dow5:
                # import copy
                # x_ori = copy.deepcopy(x)
                x, _ = self.dow5(x)  ## 18.5k --> 18.5k  [2, 1000, 1000]-->[1, 1000, 1000]




        # x4, _ = self.dow4(x)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]
        # x = self.linear_out(x4)
        # if self.use_dow5:
        #     x, _ = self.dow5(x)  ## 18.5k --> 18.5k  [2, 1000, 1000]-->[1, 1000, 1000]

        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x1,
                'x_conv2': x2,
                'x_conv3': x3,
                'x_conv4': x4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': torch.tensor([1,1,2], device=x1.features.device).float(),
                'x_conv2': torch.tensor([1,1,4], device=x1.features.device).float(),
                'x_conv3': torch.tensor([1,1,8], device=x1.features.device).float(),
                'x_conv4': torch.tensor([1,1,16], device=x1.features.device).float(),
            }
        })

        batch_dict['voxel_coords'] = x.indices
        # assert batch_dict['voxel_coords'][:, 1].max() == 0 and batch_dict['voxel_coords'][:, 1].min() == 0
        assert batch_dict['voxel_coords'][:, 0].max() == batch_size - 1 and batch_dict['voxel_coords'][:, 0].min() == 0
        assert batch_dict['voxel_coords'][:, 2].max() < 360 and batch_dict['voxel_coords'][:, 2].min() >= 0
        assert batch_dict['voxel_coords'][:, 3].max() < 360 and batch_dict['voxel_coords'][:, 3].min() >= 0
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = x.features
        return batch_dict

    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]

#for argoverse
class LION3DBackboneOneStride_Sparse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        self.group_size = model_cfg.GROUP_SIZE
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        
        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2 + 2*3

        down_scale_list = [[2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 1],
                           [1, 1, 2],
                           [1, 1, 2]
                           ]
        total_down_scale_list = [down_scale_list[0]]
        for i in range(len(down_scale_list) - 1):
            tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
            total_down_scale_list.append(tmp_dow_scale)

        assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        assert len(layer_down_scales[0]) == depths[0]
        assert len(self.layer_dim) == len(depths)

        
        self.linear_1 = LIONBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                    self.group_size[0], direction, shift=shift, operator=self.linear_operator, layer_id=0, n_layer=self.n_layer)  ##[27, 27, 32] --》 [13, 13, 32]

        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        

        # [944, 944, 16] -> [472, 472, 8]
        self.linear_2 = LIONBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                    self.group_size[1], direction, shift=shift, operator=self.linear_operator, layer_id=8, n_layer=self.n_layer)

        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)


        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = LIONBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                    self.group_size[2], direction, shift=shift, operator=self.linear_operator, layer_id=16, n_layer=self.n_layer)

        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 4] -> [236, 236, 2]
        self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                    self.group_size[3], direction, shift=shift, operator=self.linear_operator, layer_id=24, n_layer=self.n_layer)

        self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                     norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)
        
        self.dow_out = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
                                        norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_bev1 = LIONLayer(self.layer_dim[3], 1, [25, 25, 1], 512, direction=['x', 'y'], shift=shift,
                                      operator=self.linear_operator, layer_id=34, n_layer=self.n_layer)
        self.linear_bev2 = LIONLayer(self.layer_dim[3], 1, [37, 37, 1], 1024, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=36, n_layer=self.n_layer)
        self.linear_bev3 = LIONLayer(self.layer_dim[3], 1, [51, 51, 1], 2048, direction=['x', 'y'], shift=shift,
                                       operator=self.linear_operator, layer_id=38, n_layer=self.n_layer)
        

        self.num_point_features = dim

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.linear_1(x)
        x, _ = self.dow1(x)
        x = self.linear_2(x)
        x, _ = self.dow2(x)
        x = self.linear_3(x)
        x, _ = self.dow3(x)
        x = self.linear_4(x)
        x, _ = self.dow4(x)
        x = self.linear_out(x)
        
        
        x, _ = self.dow_out(x)

        x = self.linear_bev1(x)
        x = self.linear_bev2(x)
        x = self.linear_bev3(x)

        x_new = spconv.SparseConvTensor(
            features=x.features,
            indices=x.indices[:, [0, 2, 3]].type(torch.int32), #x.indices,
            spatial_shape=x.spatial_shape[1:],
            batch_size=x.batch_size
        )

        batch_dict.update({
            'encoded_spconv_tensor': x_new,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({'spatial_features_2d': x_new})

        return batch_dict


import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot_points_on_images(x_indices, images=None, save_path='output_images.png', visualization='color', draw_lines=False, line_spacing=50):
    """
    在图像上绘制点，支持两种可视化方式：
    1. 使用不同的颜色表示不同的组（默认方式）。
    2. 在点的位置绘制对应的组的 group_id。
    
    增加平行线绘制功能。
    
    参数：
    - x_indices: Tensor，形状为 [num_group, num_points_per_group, 3]
    - images: 可选，图像的列表或张量
    - save_path: 保存绘制结果的路径
    - visualization: 可视化方式，'color' 或 'group_id'
    - draw_lines: 是否绘制平行线
    - line_spacing: 平行线之间的间隔
    """

    num_group = x_indices.shape[0]
    num_points_per_group = x_indices.shape[1]

    # 将 x_indices 移动到 CPU
    x_indices = x_indices.cpu()

    # 展平 x_indices，形状为 [num_group * num_points_per_group, 3]
    x_indices_flat = x_indices.view(-1, 3)

    # 创建组标签
    group_labels = torch.arange(num_group).unsqueeze(1).expand(-1, num_points_per_group).flatten()

    # 获取所有的 image_ids 和唯一的 image_ids
    all_image_ids = x_indices_flat[:, 0].long()
    unique_image_ids = torch.unique(all_image_ids)
    num_images = len(unique_image_ids)

    # 如果选择颜色可视化方式，生成颜色列表
    if visualization == 'color':
        def get_maximally_distinct_colors(num_colors):
            hues = np.linspace(0, 1, num_colors, endpoint=False)
            indices = []
            left = 0
            right = num_colors - 1
            while left <= right:
                indices.append(left)
                left += 1
                if left <= right:
                    indices.append(right)
                    right -= 1
            hues = hues[indices]
            colors = [cm.hsv(hue) for hue in hues]
            return colors

        colors = get_maximally_distinct_colors(num_group)

    # 决定网格的大小（行和列）
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 30, rows * 20))
    axes = axes.flatten()  # 将 axes 展平，方便索引

    # 如果未提供 images，则创建空的背景
    if images is None:
        images = [None] * num_images
    else:
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        images = list(images)

    # 为每个图像绘制点
    for idx, image_id in enumerate(unique_image_ids):
        image = images[image_id] if images[image_id] is not None else None
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        image_mask = (all_image_ids == image_id)
        points_in_image = x_indices_flat[image_mask]
        group_labels_in_image = group_labels[image_mask]
        x_coords = points_in_image[:, 1].numpy()
        y_coords = points_in_image[:, 2].numpy()
        group_ids = group_labels_in_image.numpy()

        ax = axes[idx]
        if image is not None:
            ax.imshow(image)
        else:
            ax.set_xlim(0, x_coords.max())
            ax.set_ylim(0, y_coords.max())
            ax.invert_yaxis()

        if draw_lines:
            # 绘制平行于 x 和 y 轴的线
            for x in range(0, int(x_coords.max()), line_spacing):
                ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
            for y in range(0, int(y_coords.max()), line_spacing):
                ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

        if visualization == 'color':
            for group_idx in range(num_group):
                group_mask = (group_labels_in_image == group_idx)
                x = x_coords[group_mask]
                y = y_coords[group_mask]
                ax.scatter(x, y, color=colors[group_idx], label=f'Group {group_idx}', s=10)
        elif visualization == 'group_id':
            for xi, yi, gid in zip(x_coords, y_coords, group_ids):
                ax.text(xi, yi, str(gid), fontsize=6, ha='center', va='center')
        else:
            raise ValueError("Invalid visualization mode. Choose 'color' or 'group_id'.")
        
        ax.set_title(f'Image {image_id}')
        ax.axis('off')

    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.savefig(save_path)
    plt.close()
