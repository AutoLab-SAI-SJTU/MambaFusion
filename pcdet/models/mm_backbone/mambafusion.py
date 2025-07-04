import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pcdet.models.model_utils.swin_utils import PatchEmbed
from pcdet.models.model_utils.unitr_utils import MapImage2Lidar, MapLidar2Image
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned
from pcdet.models.backbones_3d.dsvt import _get_activation_fn, DSVTInputLayer
from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds
from ..vmamba.vmamba import SS2D, VSSBlock
from collections import OrderedDict
from torchvision.ops import DeformConv2d
get_inner_win_inds_cuda = ingroup_inds
import torch.nn.functional as F
from ..vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from pcdet.models.backbones_image.swin import SwinTransformer
from ..vmamba.vmamba import Backbone_VSSM
from mamba_ssm.models.mixer_seq_simple import create_block
from pcdet.models.backbones_3d.local_mamba import GlobalMamba
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv


from pcdet.models.backbones_3d.lion_backbone_one_stride import LocalMamba
from mamba_ssm import Block2 as MambaBlock
from easydict import EasyDict

from pcdet.models.backbones_image.img_neck.generalized_lss import MY_FPN

class MambaFusion(nn.Module):
    def __init__(self, model_cfg, use_map=False, **kwargs):
        super().__init__()
        self.use_cross_img = model_cfg.get('USE_CROSS_IMG', False)
        self.use_winmamba = model_cfg.get('USE_WINMAMBA', False)
        # self.use_mamba2 = model_cfg.get('USE_MAMBA2', False)
        self.use_vmamba_pretrain = model_cfg.get('USE_VMAMBA_PRETRAIN', False)
        self.use_prebackbone = model_cfg.get('USE_PREBACKBONE', False)
        self.use_mamba_inter = model_cfg.get('USE_MAMBA_INTER', False)
        self.use_mamba_inter2 = model_cfg.get('USE_MAMBA_INTER2', False)
        self.use_checkpoint_inter = model_cfg.get('USE_CHECKPOINT_INTER', False)
        self.use_checkpoint_inter2 = model_cfg.get('USE_CHECKPOINT_INTER2', False)
        self.mixed_version = model_cfg.get('MIXED_VERSION', 0)
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [256, 704])
        self.use_more_res = model_cfg.get('USE_MORE_RES', False)
        if self.use_more_res:
            res_len = len(model_cfg.out_indices) - 1
            self.res_blocks_lidar = nn.ModuleList()
            self.res_blocks_camera = nn.ModuleList()
            for i in range(res_len):
                self.res_blocks_lidar.append(nn.ModuleList([nn.LayerNorm(128),nn.LayerNorm(128)]))
                self.res_blocks_camera.append(nn.ModuleList([nn.LayerNorm(128),nn.LayerNorm(128)]))

        self.inter2_use_expand = model_cfg.get('INTER2_USE_EXPAND', False)
        self.use_fixed_mapping = model_cfg.get('USE_FIXED_MAPPING', False)
        self.use_inverse = model_cfg.get('USE_INVERSE', False)
        self.use_checkpoint_global = model_cfg.get('USE_CHECKPOINT_GLOBAL', True)
        self.use_checkpoint_local = model_cfg.get('USE_CHECKPOINT_LOCAL', True)
        # self.inter2_down_scales = model_cfg.get('INTER2_DOWN_SCALES', [[2, 2, 1], [2, 2, 1]])
        self.use_down_scale_inter2 = model_cfg.get('USE_DOWN_SCALE_INTER2', True)
        self.use_all_mamba = model_cfg.get('USE_ALL_MAMBA', False)
        self.mask_out_img = model_cfg.get('MASK_OUT_IMG', False)
        self.use_shift = model_cfg.get('USE_SHIFT', True)


        # if self.use_mamba2:
        #     self.winmamba_info = {'NAME': 'Mamba2', 'CFG': {'d_state': 128, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}
        #     self.ssm_cfg = {'layer': 'Mamba2', 'd_state': 128, 'd_conv': 4, 'expand': 2, 'headdim': 32}
        # else:
        self.localmamba_info = {'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}
        self.ssm_cfg = None
        if self.use_down_scale_inter2:
            self.inter2_down_scales = [[2, 2, 1], [2, 2, 1]]
            self.inter2_down_scales_global = [1, 2]
        else:
            self.inter2_down_scales = model_cfg.get('INTER2_DOWN_SCALES', [[1, 1, 1], [1, 1, 1]])
            self.inter2_down_scales_global = [1, 1]
        if self.use_fixed_mapping:
            assert max(max(self.inter2_down_scales)) == 1, 'Fixed mapping only support downscale 1'
        self.inter1_win_shape = model_cfg.get('INTER1_WIN_SHAPE', [13, 13, 1])
        self.inter1_win_size = model_cfg.get('INTER1_WIN_SIZE', 256)
        self.inter2_win_shape = model_cfg.get('INTER2_WIN_SHAPE', [30, 30, 1])
        self.inter2_win_size = model_cfg.get('INTER2_WIN_SIZE', 90)

        self.win_version = model_cfg.get('WIN_VERSION', 'v2')
        if self.image_shape != [256, 704]:
            model_cfg.PATCH_EMBED.image_size = self.image_shape
            model_cfg.IMAGE_INPUT_LAYER.sparse_shape = [int(self.image_shape[0] / 8), int(self.image_shape[1] / 8), 1]
            model_cfg.FUSE_BACKBONE.LIDAR2IMAGE.lidar2image_layer.sparse_shape = [int(self.image_shape[0] / 8 * 3), int(self.image_shape[1] / 8 * 3), 6]
        
        self.use_mixed_scale = model_cfg.get('USE_MIXED_SCALE', False)
        self.use_multi_scale = model_cfg.get('USE_MULTI_SCALE', False)
        self.use_multi_scalev = model_cfg.get('USE_MULTI_SCALEV', False)
        self.use_multi_scalev_down = model_cfg.get('USE_MULTI_SCALEV_DOWN', False)
        self.use_denoise = model_cfg.get('USE_DENOISE', False)
        self.use_profiler = model_cfg.get('USE_PROFILER', False)
        self.return_abs_coords = model_cfg.get('RETURN_ABS_COORDS', False)
        self.use_more_vbackbone = model_cfg.get('USE_MORE_VBACKBONE', False)

        self.use_mixed = model_cfg.get('USE_MIXED', False)
        if self.use_vmamba_pretrain and not self.use_prebackbone:
            if self.use_more_vbackbone:
                depths = (2, 2, 15)
                out_indices = [1, 2]
                fpn_model_cfg = EasyDict({
                    'IN_CHANNELS': [256, 512],  # 输入的特征图通道数
                    'OUT_CHANNELS': 256,  # 期望的输出通道数
                    'NUM_OUTS': 2,  # 生成的FPN层数
                    'START_LEVEL': 0,
                    'END_LEVEL': -1,  # 使用所有的特征层
                    'USE_BIAS': False,
                    'ALIGN_CORNERS': False
                })
                self.vssm_fpn = MY_FPN(fpn_model_cfg)
            else:
                depths = (2, 2)
                out_indices = [1] if not self.use_multi_scalev else [0, 1]
            if self.use_multi_scalev_down:
                depths = (2, 2, 2)
                out_indices += [2]
                self.vssm_multi_scalev_down_block = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
            args = {
                'norm_layer': 'ln2d',
                'patch_size': 4,
                'in_chans': 3,
                'depths': depths, # , 15, 2
                'out_indices': out_indices,
                'dims': 128,
                'ssm_d_state': 1,
                'ssm_conv_bias':False,
                'forward_type': 'v05_noz',
                'drop_path_rate': 0.2,
                'downsample_version': 'v3',
                'patchembed_version': 'v2',
            }
            # import pickle
            # with open('swin_model_cfg.pkl', 'rb') as file:
            #     swin_model_cfg = pickle.load(file)
            self.backbone_vssm = Backbone_VSSM(**args)
            # self.backbone_vssm = SwinTransformer(swin_model_cfg)
            # self.vssm_down_block = nn.Conv2d(in_channels=1344, out_channels=128, kernel_size=1, stride=1, padding=0)
            # self.backbone_vssm.register_forward_pre_hook(forward_hook_start)
            # self.backbone_vssm.register_forward_hook(forward_hook_end)
            # self.backbone_vssm.register_backward_hook(backward_hook_start)
            # self.backbone_vssm.register_backward_hook(backward_hook_end)
            
            self.vssm_down_block = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
            # self.remove_layers(self.backbone_vssm, ['outnorm2', 'outnorm3'])
            # self.remove_layers(self.backbone_vssm.layers, ['2', '3'])
        

        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        if self.use_vmamba:
            self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)

            self.vmamba_blocks = nn.ModuleList()
            depths = [2, 2]
            num_block = len(depths)
            twin_flag = [True, True]
            self.twin_flag = twin_flag
            assert len(twin_flag) == num_block, 'The length of twin_flag should be equal to num_block'
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))] 
            self.use_in_mid = False
            self.use_after = True
            self.use_conv = True

            self.lidar_fc = nn.Sequential(
                nn.Linear(128*(1+num_block), 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )

            if self.use_conv:
                for i in range(len(depths)):
                    layer = nn.LayerNorm(128)
                    layer_name = f'out_norm{i + 4}'
                    self.add_module(layer_name, layer)

                self.img_fc = nn.Sequential(
                    nn.Conv2d(in_channels=128*(1+num_block), out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            else:
                self.img_fc = nn.Sequential(
                    nn.Linear(128*3, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                )

            for i_layer in range(num_block):
                self.vmamba_blocks.append(self._make_vmamba_layer(
                    dim = 128,
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=nn.LayerNorm,
                    downsample=nn.Identity(),
                    channel_first=False,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=0,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v1dcross_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                    twin=twin_flag[i_layer],
                ))
            
        self.model_cfg = model_cfg
        self.set_info = set_info = self.model_cfg.set_info
        self.d_model = d_model = self.model_cfg.d_model
        self.nhead = nhead = self.model_cfg.nhead
        self.stage_num = stage_num = 1  # only support plain bakbone
        self.num_shifts = [2] * self.stage_num
        self.checkpoint_blocks = self.model_cfg.checkpoint_blocks
        self.image_pos_num, self.lidar_pos_num = set_info[0][-1], set_info[0][-1]
        self.accelerate = self.model_cfg.get('ACCELERATE', False)
        self.use_map = use_map

        self.image_input_layer = UniTRInputLayer(
            self.model_cfg.IMAGE_INPUT_LAYER, self.accelerate)
        self.lidar_input_layer = UniTRInputLayer(
            self.model_cfg.LIDAR_INPUT_LAYER)

        # image patch embedding
        patch_embed_cfg = self.model_cfg.PATCH_EMBED
        if (not self.use_vmamba_pretrain) and (not self.use_prebackbone):
            self.patch_embed = PatchEmbed(
                in_channels=patch_embed_cfg.in_channels,
                embed_dims=patch_embed_cfg.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_embed_cfg.patch_size,
                stride=patch_embed_cfg.patch_size,
                norm_cfg=patch_embed_cfg.norm_cfg if patch_embed_cfg.patch_norm else None
            )   
        patch_size = [patch_embed_cfg.image_size[0] // patch_embed_cfg.patch_size,
                      patch_embed_cfg.image_size[1] // patch_embed_cfg.patch_size]
        self.patch_size = patch_size
        patch_x, patch_y = torch.meshgrid(torch.arange(
            patch_size[0]), torch.arange(patch_size[1]))
        patch_z = torch.zeros((patch_size[0] * patch_size[1], 1))
        self.patch_zyx = torch.cat(
            [patch_z, patch_y.reshape(-1, 1), patch_x.reshape(-1, 1)], dim=-1).cuda()
        # patch coords with batch id
        self.patch_coords = None

        # image branch output norm
        self.out_indices = self.model_cfg.out_indices
        for i in self.out_indices:
            layer = nn.LayerNorm(d_model[-1])
            layer_name = f'out_norm{i}'
            self.add_module(layer_name, layer)

        # Sparse Regional Attention Blocks
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        layer_cfg = self.model_cfg.layer_cfg
        

        # Fuse Backbone
        fuse_cfg = self.model_cfg.get('FUSE_BACKBONE', None)
        self.fuse_on = fuse_cfg is not None
        if self.fuse_on:
            # image2lidar
            image2lidar_cfg = fuse_cfg.get('IMAGE2LIDAR', None)
            self.image2lidar_on = image2lidar_cfg is not None
            if self.image2lidar_on:
                # block range of image2lidar
                self.image2lidar_start = image2lidar_cfg.block_start
                self.image2lidar_end = image2lidar_cfg.block_end
                self.map_image2lidar_layer = MapImage2Lidar(
                    image2lidar_cfg, self.accelerate, self.use_map)
                if not (self.use_mamba_inter and not self.use_mixed):
                    self.image2lidar_input_layer = UniTRInputLayer(
                        image2lidar_cfg.image2lidar_layer)
                    self.image2lidar_pos_num = image2lidar_cfg.image2lidar_layer.set_info[0][1]
                    # encode the position of each patch from the closest point in image space
                    self.neighbor_pos_embed = PositionEmbeddingLearned(
                        2, self.d_model[-1])

            # lidar2image
            lidar2image_cfg = fuse_cfg.get('LIDAR2IMAGE', None)
            self.lidar2image_on = lidar2image_cfg is not None
            if self.lidar2image_on:
                # block range of lidar2image
                self.lidar2image_start = lidar2image_cfg.block_start
                self.lidar2image_end = lidar2image_cfg.block_end
                self.map_lidar2image_layer = MapLidar2Image(
                    lidar2image_cfg, self.accelerate, self.use_map, self.use_denoise)
                if not (self.use_mamba_inter2 and not self.use_mixed):
                    self.lidar2image_input_layer = UniTRInputLayer(
                        lidar2image_cfg.lidar2image_layer)
                    self.lidar2image_pos_num = lidar2image_cfg.lidar2image_layer.set_info[0][1]
        # new
        block_id = 0
        self.pos_embed_inter2 = nn.ModuleList()
        if self.image2lidar_on:
            self.inter_block = [i for i in range(self.image2lidar_start, self.image2lidar_end)]
        if self.lidar2image_on:
            self.inter2_block = [i for i in range(self.lidar2image_start, self.lidar2image_end)]
        if self.use_mamba_inter2:
            for block_id in self.inter2_block:
                self.image_input_layer.posembed_layers[0][block_id] = nn.Identity()
                self.lidar_input_layer.posembed_layers[0][block_id] = nn.Identity()
        if self.use_mamba_inter:
            for block_id in self.inter_block:
                self.image_input_layer.posembed_layers[0][block_id] = nn.Identity()
                self.lidar_input_layer.posembed_layers[0][block_id] = nn.Identity()
        if self.use_winmamba:
            self.image_input_layer.posembed_layers[0][0] = nn.Identity()
            self.lidar_input_layer.posembed_layers[0][0] = nn.Identity()        
        self.dsb_len = self.model_cfg.get('DSB_LEN', 1)
        self.use_offset = self.model_cfg.get('USE_OFFSET', 0)
        if self.use_offset:
            self.generate_offset = nn.ModuleList() # N*128 -> N*2 生成的offset在[-1, 1]之间
        if self.use_mamba_inter or self.use_mamba_inter2 or self.use_winmamba:
            self.hilbert_config = {'curve_template_path_rank10': '../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 
                                        'curve_template_path_rank9': '../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 
                                        'curve_template_path_rank8': '../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 
                                        'curve_template_path_rank7': '../ckpts/hilbert_template/curve_template_3d_rank_7.pth'
                                        }
            # self.hilbert_spatial_sis
            self.curve_template = {}
            self.template_on_device = False
            self.hilbert_spatial_size = {}
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 10)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 9)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 8)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_7.pth', 7)
        self.bev_size = model_cfg.get('BEV_SIZE', 360)
        self.shape_inter = [self.inter1_win_shape[-1], self.bev_size , self.bev_size ]
        self.shape_inter2 = [self.inter2_win_shape[-1], int(int(self.image_shape[1] / 8 * 3)), int(self.image_shape[0] / 8 * 3)]
        if self.mask_out_img:
            self.shape_inter2 = [self.shape_inter2[0], self.shape_inter2[1]/3,self.shape_inter2[2]/3,]
        # if self.use_multi_scalev:
        #     self.shape_inter2 = [1, 528, 192]
        self.multi_scale_norm_list = nn.ModuleList()
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_list, norm_list = [], []
            
            for i in range(num_blocks_this_stage):
                if (self.use_mamba_inter and i in self.inter_block)\
                    or (self.use_mamba_inter2 and  i in self.inter2_block):
                    if self.use_mamba_inter and i in self.inter_block:
                        self.pos_embed_inter = nn.Sequential(
                            nn.Linear(9, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128),
                            )
                    elif self.use_mamba_inter2 and  i in self.inter2_block:
                        if self.mixed_version == 0 or self.mixed_version == 2 or self.mixed_version == 3:
                            self.pos_embed_inter2.append(nn.Sequential(
                                nn.Linear(9, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                ))
                    else:
                        raise ValueError('Invalid use_mamba_inter or use_mamba_inter2')
                    if self.use_offset:
                        offset_list = nn.ModuleList()
                        for i in range(self.dsb_len):
                            offset_list.append(
                                spconv.SparseSequential(
                                    spconv.SubMConv3d(128, 2, (5, 1, 1), stride=(1, 1, 1), padding=1, bias=False),
                                    nn.BatchNorm1d(2, eps=1e-3, momentum=0.01),
                                    nn.Tanh(),)
                                )
                        self.generate_offset.append(offset_list)
                    if (self.use_multi_scalev or self.use_multi_scalev_down or self.use_multi_scale) and i != self.lidar2image_end - 1:
                        self.multi_scale_norm_list.append(nn.LayerNorm(128))

                    dsb = nn.ModuleList()
                    
                    for j in range(self.dsb_len):
                        
                        if self.use_multi_scalev:
                            # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34, use_expand=self.inter2_use_expand))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori='curve_template_rank10',
                                downsample_lvl='curve_template_rank9',
                                # downsample_lvl='curve_template_rank10',
                                # downsample_ori='curve_template_rank9',
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                        elif self.use_multi_scalev_down:
                            # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori='curve_template_rank9',
                                downsample_lvl='curve_template_rank8',
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                            
                        elif self.use_multi_scale:
                            # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori='curve_template_rank9',
                                downsample_lvl='curve_template_rank8',
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                        else:
                            if self.mixed_version == 0:
                                # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=self.use_shift,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version, use_expand=self.inter2_use_expand, use_fixed_mapping=self.use_fixed_mapping, use_inverse=self.use_inverse, use_checkpoint=self.use_checkpoint_local))
                                # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[30, 30, 1], group_size=90, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                
                                dsb.append(GlobalMamba(128, ssm_cfg=self.ssm_cfg, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=self.inter2_down_scales_global, num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8' if self.use_down_scale_inter2 else 'curve_template_rank9',
                                    down_resolution=self.use_down_scale_inter2, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32, use_checkpoint=self.use_checkpoint_global))
                            elif self.mixed_version == 1:
                                # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #                 operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #                 operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version))
                                # dsb.append(WinMamba_Block(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34, win_version=self.win_version))
                            elif self.mixed_version == 2:
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))         
                            elif self.mixed_version == 3:                    
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))     
                    
                    # if self.use_mixed:                
                    #     block_list.append(
                    #         nn.ModuleList([
                    #             dsb,
                    #             UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                    #                 dropout, activation, batch_first=True, block_id=block_id,
                    #                 dout=dmodel_this_stage, layer_cfg=layer_cfg)
                    #         ]
                    #         )
                            
                    #     )
                    # else:
                    block_list.append(dsb)
                else:

                    if self.use_winmamba and i == 0:
                        if self.inter1_win_shape[-1] != 1:
                            from pcdet.models.backbones_3d.lion_backbone_one_stride import PatchMerging3D
                            self.dow5 = PatchMerging3D(128, 128, down_scale=[1, 1, 2],
                                        norm_layer=partial(nn.LayerNorm), diffusion=True, diff_scale=0.2)
                        if self.mixed_version == 0 or self.mixed_version == 2:
                            self.pos_embed_intra = nn.Sequential(
                                nn.Linear(9, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                )
                        if self.mixed_version == 0:
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=self.use_shift,
                                                operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, use_checkpoint=self.use_checkpoint_local),
                                    GlobalMamba(128, ssm_cfg=self.ssm_cfg, norm_epsilon=1e-05, rms_norm=True, 
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                                        downsample_ori='curve_template_rank9',
                                        downsample_lvl='curve_template_rank8',
                                        # downsample_lvl='curve_template_rank9',
                                        # downsample_ori='curve_template_rank8',
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                        device='cuda', dtype=torch.float32, use_checkpoint=self.use_checkpoint_global)
                                ])
                            )
                        elif self.mixed_version == 1:      
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                ])
                            )
                        elif self.mixed_version == 2:
                            block_list.append(
                                nn.ModuleList([
                                    GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=[1, 360, 360], hilbert_config=self.hilbert_config,
                                        downsample_ori='curve_template_rank9',
                                        downsample_lvl='curve_template_rank8',
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                        device='cuda', dtype=torch.float32),
                                    GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=[1, 360, 360], hilbert_config=self.hilbert_config,
                                        downsample_ori='curve_template_rank9',
                                        downsample_lvl='curve_template_rank8',
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                        device='cuda', dtype=torch.float32)
                                ])
                            )
                        elif self.mixed_version == 3:
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),

                                ])
                            )
                    else: 
                        raise ValueError('Unexpected Parameters')                  
                        # block_list.append(
                        #     UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                        #             dropout, activation, batch_first=True, block_id=block_id,
                        #             dout=dmodel_this_stage, layer_cfg=layer_cfg)
                        # )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
                block_id += 1
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(
                f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))
            if layer_cfg.get('split_residual', False):
                # use different norm for lidar and image
                lidar_norm_list = [nn.LayerNorm(
                    dmodel_this_stage) for _ in range(num_blocks_this_stage)]
                self.__setattr__(
                    f'lidar_residual_norm_stage_{stage_id}', nn.ModuleList(lidar_norm_list))



        self._reset_parameters()
        # self.register_hooks(self) # 注册钩子
    def remove_layers(self, att_name,  layers_to_remove):
        for layer_name in layers_to_remove:
            if hasattr(att_name, layer_name):
                delattr(att_name, layer_name)
            else:
                print(f"Layer {layer_name} not found in Backbone_VSSM")
    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    def register_hooks(self, module_name):
        def create_hook(module_name):
            def check_param(module, input, output):
                if module_name in self.used_params:
                    raise ValueError(f"Parameter {module_name} is used more than once")
                else:
                    self.used_params.add(module_name)
            return check_param
        def is_leaf_module(module):
            return len(list(module.children())) == 0
        # 注册钩子
        self.all_params = set()
        self.used_params = set()
        for name, module in module_name.named_modules():           
            if len(list(module.parameters())) > 0 and is_leaf_module(module):  # 只对有参数的模块注册钩子
                self.all_params.add(name)
                module.register_forward_hook(create_hook(name))
    @staticmethod
    def _make_vmamba_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        twin=False,
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                cross_dim=dim,
                drop_path= drop_path[d % int(depth/2)] if twin else drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))
        if twin:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks[:int(len(blocks)/2)]),
                blocks2=nn.Sequential(*blocks[int(len(blocks)/2):]),
            ))
        elif cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
    def forward_stage2(self, block, output_sparse, lidar2image_coor, batch_dict, i, fixed_mapping_list=None, voxel_num=None):

        for j, b in enumerate(block):
            if self.use_more_res:
                res_output = output_sparse.features.clone()
            if isinstance(b, LocalMamba):
                if self.use_fixed_mapping:
                    output_sparse, fixed_mapping_list = b(output_sparse, fixed_mapping_list)
                else:
                    output_sparse = b(output_sparse)

                if self.use_more_res:
                    output = output_sparse.features
                    # output = self.res_blocks[i - 1][j](output + res_output)
                    output = torch.cat([self.res_blocks_lidar[i - 1][j](output[:voxel_num] + res_output[:voxel_num]), 
                                        self.res_blocks_camera[i - 1][j](output[voxel_num:] + res_output[voxel_num:])])
                    output_sparse = replace_feature(output_sparse, output)
                else:
                    if j == len(block) - 1:
                        output = output_sparse.features
            elif isinstance(b, GlobalMamba):
                output, _ = b(output_sparse.features, lidar2image_coor, batch_dict['batch_size'],
                                        self.shape_inter2, self.curve_template, self.hilbert_spatial_size,
                                        self.pos_embed_inter2[i - self.lidar2image_start], 0, False)
                # if not (j == len(block) - 1 and i == self.lidar2image_end - 1):
                if self.use_more_res:
                    # output = self.res_blocks[i - 1][j](output + res_output)
                    output = torch.cat([self.res_blocks_lidar[i - 1][j](output[:voxel_num] + res_output[:voxel_num]), 
                                        self.res_blocks_camera[i - 1][j](output[voxel_num:] + res_output[voxel_num:])])
                    output_sparse = replace_feature(output_sparse, output)
                else:
                    if not (j == len(block) - 1 and i == self.lidar2image_end - 1):
                        output_sparse = replace_feature(output_sparse, output)

                
            else:
                raise ValueError("Block type not supported")

        # if i == self.lidar2image_end - 1:
        #     return output_extended[:ori_len], output_sparse
        if self.use_fixed_mapping:
            return output, output_sparse, fixed_mapping_list
        else:
            return output, output_sparse
    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - camera_imgs (Tensor[float]): multi view images, shape of (B, N, C, H, W),
                    where N is the number of image views.
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - image_features (Tensor[float]):
        '''
        # lidar(3d) and image(2d) preprocess
        self.used_params = set()
        batch_dict['use_all_mamba'] = self.use_all_mamba
        if self.use_all_mamba:
            multi_feat, voxel_info, patch_info = self._input_preprocess2(batch_dict)
            multi_pos_embed_list = None
        else:
            if self.use_cross_img:
                multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list, sweeps_imgs, num_set_ponit = self._input_preprocess(
                    batch_dict)
            else:
                multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list = self._input_preprocess(
                    batch_dict)
        # lidar(3d) and image(3d) preprocess multi_set_voxel_inds_list[0][1][:, num_set_ponit:,: ] - batch_dict['voxel_num']
        if self.image2lidar_on: # 将image的feature映射到lidar的feature 将图像数据预处理为激光雷达特征，以增强点云的特征表示
            if self.use_mamba_inter:
                image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list, image2lidar_coor = self._image2lidar_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
                if self.shape_inter[0] == 2:
                    image2lidar_coor[:batch_dict['voxel_num'], 1] = 1
            else:
                image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list = self._image2lidar_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
        # lidar(2d) and image(2d) preprocess
        if self.lidar2image_on: # 将lidar的feature映射到image的feature 将激光雷达数据预处理为图像特征，以增强图像的特征表示
            if self.use_mamba_inter2:
                lidar2image_view_num = 6
                if self.use_multi_scale:
                    lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, lidar2image_coor, lidar2image_coords_bzyx_list = self._lidar2image_preprocess(
                        batch_dict, multi_feat, multi_pos_embed_list)
                    for i in range(len(lidar2image_coords_bzyx_list)):
                        lidar2image_coords_bzyx_list[i][:, 0] = lidar2image_coords_bzyx_list[i][:, 0] * lidar2image_view_num + lidar2image_coords_bzyx_list[i][:, 1]
                        lidar2image_coords_bzyx_list[i][:, 1] = 0

                else:
                    lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, lidar2image_coor = self._lidar2image_preprocess(
                        batch_dict, multi_feat, multi_pos_embed_list)
                if self.win_version == 'v4':
                    pass
                else:
                    lidar2image_coor[:, 0] = lidar2image_coor[:, 0] * lidar2image_view_num + lidar2image_coor[:, 1]
                    lidar2image_coor[:, 1] = 0
                    if self.shape_inter2[0] == 2:
                        lidar2image_coor[:batch_dict['voxel_num'], 1] = 1
            else:
                lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list = self._lidar2image_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
        output = multi_feat # torch.Size([num_of_voxel + num_of_patch, 128])
        block_id = 0
        voxel_num = batch_dict['voxel_num'] # num_of_voxels
        
        batch_dict['image_features'] = []
        if self.use_vmamba:
            voxel_pos_embeding = self.lidar_pos_embed_layer(batch_dict['voxel_coords'][:, 1:].float())
            lidar2image = batch_dict['lidar2image'].clone()
            lidar2image = lidar2image.view(batch_dict['batch_size'] * 6, 16)
            lidar2image = lidar2image.repeat_interleave(2816, dim=0) # [33792, 16]
            img_patch_coords = self.patch_coords.clone()
            img_patch_coords[:, 0] = img_patch_coords[:, 0] % 6 # [33792, 4]
            img_pos_embeding = self.img_pos_embed_layer(torch.cat([img_patch_coords, lidar2image], dim=1).float())
            multi_pos_embeding = torch.cat([voxel_pos_embeding, img_pos_embeding], dim=0) # 
            # img_pos_embeding = 
        if self.use_mamba_inter or self.use_mamba_inter2 or self.use_winmamba and not self.template_on_device:
            with torch.no_grad():
                for name, _ in self.curve_template.items():
                    self.curve_template[name] = self.curve_template[name].to(output.device)
            self.template_on_device = True
        # block forward
        for stage_id in range(self.stage_num): # 1
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(
                f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)): # 4 (1 2 1)
                block = block_layers[i]
                residual = output.clone() # torch.Size([num_of_voxel + num_of_patch, 128])                 

                if self.image2lidar_on and i >= self.image2lidar_start and i < self.image2lidar_end: # i == 3 模态间注意力机制
                    if self.use_mamba_inter:
                        
                        if self.use_mixed:
                            output, _ = block[0](output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter,
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                            output = block[1](output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                            
                        else:
                            if self.use_checkpoint_inter:
                                output, _ = checkpoint(block, output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter, 
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                            else:
                                output, _ = block(output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter,
                                        self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                    else:
                        output = block(output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    if self.use_vmamba and self.use_in_mid:
                        output = self.do_vmamba_inter(output, voxel_num, batch_dict, block_id)
                elif self.lidar2image_on and i >= self.lidar2image_start and i < self.lidar2image_end: #? i == 1 or 2  可能有两个不同的阶段需要进行激光雷达到图像的特征映射操作
                    if self.use_mamba_inter2:
                        if self.use_mixed:
                            output, _ = block[0](output, lidar2image_coor, batch_dict['batch_size'], self.shape_inter2,
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter2[i - self.lidar2image_start], 0, False)
                            output = block[1](output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                        elif self.use_mixed_scale:
                            if i == self.lidar2image_start:
                                output_extended = output
                                ori_len = output.shape[0]
                                if self.use_multi_scalev_down:
                                    lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_down_features_coords']], dim=0)   
                                    output_extended = torch.cat([output_extended, batch_dict['multi_scalev_down_features']], dim=0)
                                if self.use_multi_scale:
                                    use_conv_list = ['x_conv3']
                                    multi_scale_3d_features = batch_dict['multi_scale_3d_features']
                                    extends = torch.cat([multi_scale_3d_features[conv_name].features for conv_name in use_conv_list], dim=1)
                                    lidar2image_coor = torch.cat([lidar2image_coor, torch.cat(lidar2image_coords_bzyx_list)], dim=0)
                                    output_extended = torch.cat([output_extended, extends], dim=0)
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)

                        elif self.use_multi_scalev_down:
                            if i == self.lidar2image_start:
                                lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_down_features_coords']], dim=0)
                                ori_len = output.shape[0]
                                output_extended = torch.cat([output, batch_dict['multi_scalev_down_features']], dim=0)
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)

                        elif self.use_multi_scalev:
                            if i == self.lidar2image_start:
                                lidar2image_coor[:, -1] *= 2
                                lidar2image_coor[:, -2] *= 2
                                lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_features_coords']], dim=0)
                                ori_len = output.shape[0]
                                output_extended = torch.cat([output, batch_dict['multi_scalev_features']], dim=0)
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=[1, self.shape_inter2[1] * 2, self.shape_inter2[2] * 2],
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )      
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            if self.use_profiler:
                                with torch.profiler.record_function("block_{}".format(i)):
                                    output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)
                            else:
                                output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)
                                    
                        elif self.use_multi_scale:
                            if i == self.lidar2image_start:
                                use_conv_list = ['x_conv3']
                                multi_scale_3d_features = batch_dict['multi_scale_3d_features']
                                extends = torch.cat([multi_scale_3d_features[conv_name].features for conv_name in use_conv_list], dim=1)
                                lidar2image_coor = torch.cat([lidar2image_coor, torch.cat(lidar2image_coords_bzyx_list)], dim=0)
                                output_extended = torch.cat([output, extends], dim=0)
                                ori_len = output.shape[0]
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()

                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)


                        else:
                            if i == self.lidar2image_start:
                                if self.mask_out_img:
                                    out_put_ori = output.clone()
                                    selected_indices = (lidar2image_coor[:, 2] >= 88) & (lidar2image_coor[:, 2] < 176) & (lidar2image_coor[:, 3] >= 32) & (lidar2image_coor[:, 3] < 64)
                                    lidar2image_coor = lidar2image_coor[selected_indices]
                                    output = output[selected_indices]
                                    lidar2image_coor[:, 2] -= 88
                                    lidar2image_coor[:, 3] -= 32
                                    output_sparse = spconv.SparseConvTensor(
                                        features=output,
                                        indices=lidar2image_coor.int(),
                                        spatial_shape=self.shape_inter2,
                                        batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                    )
                                else:
                                    output_sparse = spconv.SparseConvTensor(
                                        features=output,
                                        indices=lidar2image_coor.int(),
                                        spatial_shape=self.shape_inter2,
                                        batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                    )
                            else: # 由于进行了norm以及融合了residual，所以这里需要将output_sparse的features替换为output
                                if self.mask_out_img:
                                    output_sparse = replace_feature(output_sparse, output[selected_indices])
                                else:
                                    output_sparse = replace_feature(output_sparse, output)
                            
                            if self.use_fixed_mapping:
                                if i == self.lidar2image_start:
                                    fixed_mapping_list = None
                                output, output_sparse, fixed_mapping_list = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i, fixed_mapping_list)
                            else:
                                output, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor.int(), batch_dict, i, voxel_num=voxel_num)
                            if self.mask_out_img:
                                out_put_ori[selected_indices] = output
                                output = out_put_ori



                    else:
                        output = block(output, lidar2image_inds_list[stage_id], lidar2image_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    if self.use_vmamba and self.use_in_mid:
                        output = self.do_vmamba_inter(output, voxel_num, batch_dict, block_id)
                else: # i == 0 模态内注意力机制 反正也是分开进行的，所以这里可以分开？
                    if self.use_vmamba and self.use_in_mid:
                        output = output + multi_pos_embeding

                    if self.use_winmamba:

                        image_coords = batch_dict['patch_coords'].clone()
                        image_coords[:, 0] = image_coords[:, 0] + batch_dict['batch_size']
                        indices = torch.cat([batch_dict['voxel_coords'], image_coords], dim=0)
                        h_size = self.inter1_win_shape[-1]
                        output_sparse = spconv.SparseConvTensor(
                            features=output,
                            indices=indices.int(),
                            spatial_shape=[h_size, self.bev_size , self.bev_size ],
                            batch_size=batch_dict['batch_size'] * 7
                        )
                        for j, b in enumerate(block):
                            if isinstance(b, LocalMamba):
                                output_sparse = b(output_sparse)
                                if j == len(block) - 1:
                                    output = output_sparse.features
                            elif isinstance(b, GlobalMamba):
                                output, _ = b(output_sparse.features, indices, batch_dict['batch_size'], [h_size, self.bev_size , self.bev_size ],
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_intra, 0, False)
                                if not (j == len(block) - 1):
                                    output_sparse = replace_feature(output_sparse, output)
                            else:
                                raise ValueError("Block type not supported")
                            # if j % 2 == 0:
                            #     output_sparse = b(output_sparse)
                            # else:
                            #     output, _ = b(output_sparse.features, indices, batch_dict['batch_size'], [1, 360, 360],
                            #         self.curve_template, self.hilbert_spatial_size, self.pos_embed_intra, 0, False)


                    else:
                        output = block(output, multi_set_voxel_inds_list[stage_id], multi_set_voxel_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)

                        
                    # if self.use_vmamba and self.use_in_mid:
                    #     output = self.do_vmamba_intra(output, voxel_num, batch_dict, block_id)
                # use different norm for lidar and image
                if self.model_cfg.layer_cfg.get('split_residual', False): # True 分别进行两个模态的norm
                    if 'output_extended' in locals():
                        output = output_extended[:ori_len]
                        if i != self.lidar2image_end - 1:
                            extended = self.multi_scale_norm_list[i-self.lidar2image_start](output_extended[ori_len:] + residual_extended)
                    output = torch.cat([self.__getattr__(f'lidar_residual_norm_stage_{stage_id}')[i](output[:voxel_num] + residual[:voxel_num]),
                                       residual_norm_layers[i](output[voxel_num:] + residual[voxel_num:])], dim=0)
                    if 'output_extended' in locals() and i != self.lidar2image_end - 1:
                        output_extended = torch.cat([output, extended], dim=0)
                    
                else:
                    output = residual_norm_layers[i](output + residual)
                block_id += 1
                # recover image feature shape
                if i in self.out_indices: # []
                    if i == self.out_indices[-1] and self.use_vmamba and self.use_after:

                        # output_new = output + multi_pos_embeding
                        output_new_self_list = []
                        output_new_cross_list = []
                        for block_id in range(len(self.vmamba_blocks)):
                            if self.twin_flag[block_id]:
                                output_new_self_list.append(self.do_vmamba_intra(output, voxel_num, batch_dict, block_id, multi_pos_embeding))
                            else:
                                output_new_cross_list.append(self.do_vmamba_inter(output, voxel_num, batch_dict, block_id, multi_pos_embeding))
                        # output_new_self = self.do_vmamba_intra(output, voxel_num, batch_dict, 0, multi_pos_embeding)
                        # output_new_cross = self.do_vmamba_inter(output, voxel_num, batch_dict, 1, multi_pos_embeding)
                        # output_new_self2 = self.do_vmamba_intra(output_new_self, voxel_num, batch_dict, 2, multi_pos_embeding)
                        # output_new_cross2 = self.do_vmamba_inter(output_new_self, voxel_num, batch_dict, 3, multi_pos_embeding)
                        # output_new = self.do_vmamba_intra(output_new, voxel_num, batch_dict, 2)
                        # output_new = self.do_vmamba_inter(output_new, voxel_num, batch_dict, 3)

                        # lidar_part = torch.cat([output[:voxel_num], output_new_self[:voxel_num], output_new_cross[:voxel_num], output_new_self2[:voxel_num], output_new_cross2[:voxel_num]], dim=-1).contiguous()
                        lidar_part = torch.cat([output[:voxel_num]] + [output_new_self[:voxel_num] for output_new_self in output_new_self_list] + [output_new_cross[:voxel_num] for output_new_cross in output_new_cross_list], dim=-1).contiguous()
                        lidar_part = self.lidar_fc(lidar_part)

                        if self.use_conv:
                            batch_spatial_features_new_self_list = [self._recover_image(pillar_features=output_new_self_list[block_id][voxel_num:],
                                                                            coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i+1+block_id)
                                                                            for block_id in range(len(output_new_self_list))]
                            batch_spatial_features_new_cross_list = [self._recover_image(pillar_features=output_new_cross_list[block_id][voxel_num:],
                                                                            coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i+1+block_id + len(output_new_self_list))
                                                                            for block_id in range(len(output_new_cross_list))]
                            # batch_spatial_features_new_self = self._recover_image(pillar_features=output_new_self[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 1)
                            # batch_spatial_features_new_cross = self._recover_image(pillar_features=output_new_cross[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 2)
                            # batch_spatial_features_new_self2 = self._recover_image(pillar_features=output_new_self2[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 3)
                            # batch_spatial_features_new_cross2 = self._recover_image(pillar_features=output_new_cross2[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 4)

                            batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                        coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)

                            # batch_spatial_features = torch.cat([batch_spatial_features, batch_spatial_features_new_self, batch_spatial_features_new_cross], dim=1) # [24, 128, 32, 88]
                            # batch_spatial_features = torch.cat([batch_spatial_features, batch_spatial_features_new_self, batch_spatial_features_new_cross, batch_spatial_features_new_self2, batch_spatial_features_new_cross2], dim=1) # [24, 128, 32, 88]
                            batch_spatial_features = torch.cat([batch_spatial_features] + batch_spatial_features_new_self_list + batch_spatial_features_new_cross_list, dim=1) # [24, 128, 32, 88]
                            batch_spatial_features = self.img_fc(batch_spatial_features)
                            image_part = batch_spatial_features.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                            output = torch.cat([lidar_part, image_part], dim=0)
                        else:
                            image_part = torch.cat([output[voxel_num:], output_new_self[voxel_num:], output_new_cross[voxel_num:]], dim=-1).contiguous()
                            image_part = self.img_fc(image_part)
                            output = torch.cat([lidar_part, image_part], dim=0)
                            batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                    coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                        
                        
                    else:
                        batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:], # 前面分开norm，这里+1？
                                                                    coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                    batch_dict['image_features'].append(batch_spatial_features) # [24, 128, 32, 88]
        if self.use_winmamba and h_size != 1:
            lidar_sparse = spconv.SparseConvTensor(
                            features=output[:voxel_num],
                            indices=batch_dict['voxel_coords'],
                            spatial_shape=[h_size, 360, 360],
                            batch_size=batch_dict['batch_size']
                        )
            lidar_sparse, _ = self.dow5(lidar_sparse)
            batch_dict['pillar_features'] = lidar_sparse.features
            batch_dict['voxel_coords'] = lidar_sparse.indices
        else:
            batch_dict['pillar_features'] = batch_dict['voxel_features'] = output[:voxel_num]
            batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict
    def do_vmamba_inter(self, output, voxel_num, batch_dict, block_id, embeding=None):
        img_num_per_frame = 16896
        img_part = output[voxel_num:, :]
        img_part = img_part.reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        point_part = output[:voxel_num, :]
        new_img_part_frame_list = []
        new_point_part_frame_list = []
        if embeding is not None:
            point_embeding = embeding[:voxel_num]
            img_embeding = embeding[voxel_num:]
        for batch_id in range(batch_dict['batch_size']):
            current_index = batch_dict['voxel_coords'][:, 0] == batch_id
            point_part_frame = point_part[current_index]
            multi_part_frame = torch.cat([point_part_frame, img_part[batch_id]], dim=0).contiguous()
            if embeding is not None:
                point_part_frame_with_embed = point_part_frame + point_embeding[current_index]
                img_part_frame_with_embed = img_part[batch_id] + img_embeding[batch_id]
                multi_part_frame_with_embed = torch.cat([point_part_frame_with_embed, img_part_frame_with_embed], dim=0).contiguous()
                multi_part_frame_new = self.vmamba_blocks[block_id]((multi_part_frame, multi_part_frame_with_embed)).squeeze()
            else:
                multi_part_frame_new = self.vmamba_blocks[block_id](multi_part_frame).squeeze()
            new_point_part_frame = multi_part_frame_new[:point_part_frame.shape[0]]
            new_img_part_frame = multi_part_frame_new[point_part_frame.shape[0]:]
            new_img_part_frame_list.append(new_img_part_frame)
            new_point_part_frame_list.append(new_point_part_frame)
            
        output_new = torch.cat(new_point_part_frame_list + new_img_part_frame_list, dim=0)
        return output_new
    def do_vmamba_intra(self, output, voxel_num, batch_dict, block_id, embeding=None):
        img_num_per_frame = 16896
        img_part = output[voxel_num:, :]
        img_part = img_part.reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        point_part = output[:voxel_num, :]
        new_img_part_frame_list = []
        new_point_part_frame_list = []
        len_vmamba = len(self.vmamba_blocks[block_id])
        if embeding is not None:
            point_embeding = embeding[:voxel_num]
            img_embeding = embeding[voxel_num:].reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        for batch_id in range(batch_dict['batch_size']):
            current_index = batch_dict['voxel_coords'][:, 0] == batch_id
            point_part_frame = point_part[current_index] # [num_of_voxel, 128]
            if embeding is not None:
                point_part_frame_with_embed = point_part_frame + point_embeding[current_index]
                new_point_part_frame = self.vmamba_blocks[block_id].blocks1((point_part_frame, point_part_frame_with_embed)).squeeze()
                img_part_frame_with_embed = img_part[batch_id] + img_embeding[batch_id]
                new_img_part_frame =  self.vmamba_blocks[block_id].blocks2((img_part[batch_id], img_part_frame_with_embed)).squeeze()
            else:
                new_point_part_frame = self.vmamba_blocks[block_id].blocks1(point_part_frame).squeeze()
                new_img_part_frame =  self.vmamba_blocks[block_id].blocks2(img_part[batch_id]).squeeze()
            new_img_part_frame_list.append(new_img_part_frame)
            new_point_part_frame_list.append(new_point_part_frame)
            
        output_new = torch.cat(new_point_part_frame_list + new_img_part_frame_list, dim=0) # [num_of_voxel+num_of_patch, 128]
        return output_new
    def _input_preprocess(self, batch_dict):
        # image branch
        imgs = batch_dict['camera_imgs'] # B, N, C, H, W
        B, N, C, H, W = imgs.shape  # 2, 6, 3, 256, 704
        imgs = imgs.view(B * N, C, H, W) # 12, 3, 256, 704
        if self.use_prebackbone:
            imgs = batch_dict['backbone_features']
            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        elif self.use_vmamba_pretrain:

            imgs_list = checkpoint(self.backbone_vssm, imgs, use_reentrant=False)
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features'] = imgs_list[0]
                batch_dict['multi_scalev_features'] = batch_dict['multi_scalev_features'].permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_features_coords'] = None
            if self.use_multi_scalev_down:
                down_imgs = imgs_list[-1]
                down_imgs = self.vssm_multi_scalev_down_block(down_imgs)
                batch_dict['multi_scalev_down_features'] = down_imgs.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_down_features_coords'] = None
            imgs = imgs_list[1] if self.use_multi_scalev else imgs_list[0]
            imgs = self.vssm_down_block(imgs) # [6, 128, 32, 88]

            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        else:
            imgs, hw_shape = self.patch_embed(imgs)  # [12, 2816, 128] (32, 88) 将图像转换为patch 2816个patch，每个patch 128维
        if self.use_cross_img:
            sweeps_imgs = batch_dict['img_sweeps']
            T = batch_dict['img_sweeps'].shape[2]
            sweeps_imgs = sweeps_imgs.view(B * N * T, C, H, W)
            with torch.no_grad():
                if self.use_vmamba_pretrain:
                    sweeps_imgs = self.backbone_vssm(sweeps_imgs)[0]
                    # sweeps_imgs = checkpoint(self.backbone_vssm, sweeps_imgs)[0]
                    sweeps_imgs = self.vssm_down_block(sweeps_imgs) 
                    sweeps_imgs = sweeps_imgs.permute(0, 2, 3, 1).contiguous().view(B * N, T, -1, 128) # [6, 2, 2816, 128]
                else:
                    sweeps_imgs, _ = self.patch_embed(sweeps_imgs) # [12, 2816, 128])
                    sweeps_imgs = sweeps_imgs.view(B * N, T, sweeps_imgs.shape[1], 128) # [6, 2, 2816, 128]


        batch_dict['hw_shape'] = hw_shape

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1]) # [num_of_patch, 128]
        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone() # [num_of_patch, 4] image_id z y x
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone()[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
            if self.use_multi_scalev:
                new_hw_shape = [size * 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords[:, -1] += new_hw_shape[0]
                self.new_patch_coords[:, -2] += new_hw_shape[1]
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                new_hw_shape = [size // 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords_down = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords_down[:, -2:] *= 2
                self.new_patch_coords_down[:, -1] += new_hw_shape[0] * 2
                self.new_patch_coords_down[:, -2] += new_hw_shape[1] * 2
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        patch_info = self.image_input_layer(batch_dict)
        patch_feat = batch_dict['patch_features'] # [num_of_patch, 128]
        patch_set_voxel_inds_list = [[patch_info[f'set_voxel_inds_stage{s}_shift{i}'] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set, 90]
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_set_voxel_masks_list = [[patch_info[f'set_voxel_mask_stage{s}_shift{i}'] # 重复voxel的mask 每个元素 [2, num_of_set, 90]
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_pos_embed_list = [[[patch_info[f'pos_embed_stage{s}_block{b}_shift{i}'] # 每个set里每个block的pos_embed 每个元素 [num_of_voxel, 128]
                                  for i in range(self.num_shifts[s])] for b in range(self.image_pos_num)] for s in range(len(self.set_info))]

        # lidar branch
        voxel_info = self.lidar_input_layer(batch_dict)
        voxel_feat = batch_dict['voxel_features']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
                                for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}']
                                 for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                            for i in range(self.num_shifts[s])] for b in range(self.lidar_pos_num)] for s in range(len(self.set_info))]

        # multi-modality parallel
        voxel_num = voxel_feat.shape[0] # num_of_voxels
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0) # [num_of_voxels+num_of_patch, 128]
        
        multi_set_voxel_inds_list = [[torch.cat([set_voxel_inds_list[s][i], patch_set_voxel_inds_list[s][i]+voxel_num], dim=1)
                                        for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set(voxel)+num_of_set(patch), 90]
        multi_set_voxel_masks_list = [[torch.cat([set_voxel_masks_list[s][i], patch_set_voxel_masks_list[s][i]], dim=1)
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 重复voxel的mask 每个元素 [2, num_of_set(voxel)+num_of_set(patch), 90]
        multi_pos_embed_list = []
        for s in range(len(self.set_info)): # 1
            block_pos_embed_list = []
            for b in range(self.set_info[s][1]): # 4
                shift_pos_embed_list = []
                for i in range(self.num_shifts[s]): # 2
                    if b < self.lidar_pos_num and b < self.image_pos_num: # ? 好像必然是这个branch
                        if (self.use_mamba_inter2 and b in self.inter2_block ) or (self.use_mamba_inter and b in self.inter_block) or (self.use_winmamba and b == 0):
                            shift_pos_embed_list.append([])
                        else:
                            shift_pos_embed_list.append(
                                torch.cat([pos_embed_list[s][b][i], patch_pos_embed_list[s][b][i]], dim=0)) # [num_of_voxel+num_of_patch, 128]
                    elif b < self.lidar_pos_num and b >= self.image_pos_num:
                        shift_pos_embed_list.append(pos_embed_list[s][b][i])
                    elif b >= self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            patch_pos_embed_list[s][b][i])
                    else:
                        raise NotImplementedError
                block_pos_embed_list.append(shift_pos_embed_list)
            multi_pos_embed_list.append(block_pos_embed_list)
        if self.use_cross_img: # multi_set_voxel_inds_list[0][1][:, set_voxel_inds_list[0][0].shape[1]:,: ] - batch_dict['voxel_num']
            return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list, sweeps_imgs, [set_voxel_inds_list[0][i].shape[1] for i in range(self.num_shifts[0])]
        else:
            return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list

    def _input_preprocess2(self, batch_dict):
        # image branch
        imgs = batch_dict['camera_imgs'] # B, N, C, H, W
        B, N, C, H, W = imgs.shape  # 2, 6, 3, 256, 704
        imgs = imgs.view(B * N, C, H, W) # 12, 3, 256, 704
        if self.use_prebackbone:
            imgs = batch_dict['backbone_features']
            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        elif self.use_vmamba_pretrain:

            if self.training:
                imgs_list = checkpoint(self.backbone_vssm, imgs, use_reentrant=False)
            else:
                imgs_list = self.backbone_vssm(imgs)
            if self.use_more_vbackbone:
                imgs_list = self.vssm_fpn(imgs_list)
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features'] = imgs_list[0]
                batch_dict['multi_scalev_features'] = batch_dict['multi_scalev_features'].permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_features_coords'] = None
            if self.use_multi_scalev_down:
                down_imgs = imgs_list[-1]
                down_imgs = self.vssm_multi_scalev_down_block(down_imgs)
                batch_dict['multi_scalev_down_features'] = down_imgs.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_down_features_coords'] = None
            imgs = imgs_list[1] if self.use_multi_scalev else imgs_list[0]
            imgs = self.vssm_down_block(imgs) # [6, 128, 32, 88]

            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        else:
            imgs, hw_shape = self.patch_embed(imgs)  # [12, 2816, 128] (32, 88) 将图像转换为patch 2816个patch，每个patch 128维

        batch_dict['hw_shape'] = hw_shape

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1]) # [num_of_patch, 128]
        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone() # [num_of_patch, 4] image_id z y x
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone()[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
            if self.use_multi_scalev:
                new_hw_shape = [size * 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords[:, -1] += new_hw_shape[0]
                self.new_patch_coords[:, -2] += new_hw_shape[1]
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                new_hw_shape = [size // 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords_down = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords_down[:, -2:] *= 2
                self.new_patch_coords_down[:, -1] += new_hw_shape[0] * 2
                self.new_patch_coords_down[:, -2] += new_hw_shape[1] * 2
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        patch_info = {f'voxel_coors_stage0': batch_dict['patch_coords']}
        patch_feat = batch_dict['patch_features'] # [num_of_patch, 128]

        # lidar branch
        voxel_info = {f'voxel_coors_stage0': batch_dict['voxel_coords']}
        voxel_feat = batch_dict['voxel_features']

        # multi-modality parallel
        voxel_num = voxel_feat.shape[0] # num_of_voxels
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0) # [num_of_voxels+num_of_patch, 128]

        return multi_feat, voxel_info, patch_info


    def _image2lidar_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        N = batch_dict['camera_imgs'].shape[1] # 6
        voxel_num = batch_dict['voxel_num'] # num_of_voxels
        image2lidar_coords_zyx, nearest_dist = self.map_image2lidar_layer(
            batch_dict) # [num_of_patch, 3] [num_of_patch, 1] 最近的3D点的坐标和距离
        image2lidar_coords_bzyx = torch.cat(
            [batch_dict['patch_coords'][:, :1].clone(), image2lidar_coords_zyx], dim=1) # [num_of_patch, 4] image_id z y x
        image2lidar_coords_bzyx[:, 0] = image2lidar_coords_bzyx[:, 0] // N # torch.Size([33792, 4]) batch_id z y x
        if not (self.use_mamba_inter and not self.use_mixed):
            image2lidar_batch_dict = {}
            image2lidar_batch_dict['voxel_features'] = multi_feat.clone() # torch.Size([num_of_voxel + num_of_patch, 128])
            image2lidar_batch_dict['voxel_coords'] = torch.cat( # torch.Size([num_of_voxel + num_of_patch, 4])
                [batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)

            image2lidar_info = self.image2lidar_input_layer(image2lidar_batch_dict) # 得到增强点云的信息 
            image2lidar_inds_list = [[image2lidar_info[f'set_voxel_inds_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set, 90]
            image2lidar_masks_list = [[image2lidar_info[f'set_voxel_mask_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 重复voxel的mask 每个元素 [2, num_of_set, 90]
            image2lidar_pos_embed_list = [[[image2lidar_info[f'pos_embed_stage{s}_block{b}_shift{i}']   # 每个set里每个block的pos_embed 每个元素 [num_of_voxel, 128]
                                            for i in range(self.num_shifts[s])] for b in range(self.image2lidar_pos_num)] for s in range(len(self.set_info))]
            image2lidar_neighbor_pos_embed = self.neighbor_pos_embed(nearest_dist) # [num_of_patch, 128]

            for b in range(self.image2lidar_start, self.image2lidar_end): # 3, 4
                for i in range(self.num_shifts[0]): # 2
                    image2lidar_pos_embed_list[0][b - 
                                                self.image2lidar_start][i][voxel_num:] += image2lidar_neighbor_pos_embed # [num_of_patch, 128]
                    multi_pos_embed_list[0][b][i] += image2lidar_pos_embed_list[0][b -
                                                                                self.image2lidar_start][i] # [num_of_voxel + num_of_patch, 128]
        if self.use_mamba_inter:
            if self.use_mixed:
                return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list, image2lidar_batch_dict['voxel_coords']
            else:
                return None, None, multi_pos_embed_list, torch.cat([batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)
        else:

            return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list

    def _lidar2image_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        N = batch_dict['camera_imgs'].shape[1] # 6
        hw_shape = batch_dict['hw_shape'] # [32, 88]
        lidar2image_coords_zyx, lidar2image_coords_bzyx_list = self.map_lidar2image_layer(batch_dict, self.use_multi_scale) # [num_of_voxel, 3] view_idx, x, y 点云投影到图像上的坐标
        lidar2image_coords_bzyx = torch.cat(
            [batch_dict['voxel_coords'][:, :1].clone(), lidar2image_coords_zyx], dim=1) # torch.Size([num_of_voxel, 4])

        multiview_coords = batch_dict['patch_coords'].clone() # torch.Size([num_of_patch, 4])
        multiview_coords[:, 0] = batch_dict['patch_coords'][:, 0] // N # 得到batch_id
        multiview_coords[:, 1] = batch_dict['patch_coords'][:, 0] % N # 得到view_id
        multiview_coords[:, 2] += hw_shape[1] # 得到y
        multiview_coords[:, 3] += hw_shape[0] # 得到x

        if not (self.use_mamba_inter2 and not self.use_mixed):

            lidar2image_batch_dict = {}
            lidar2image_batch_dict['voxel_features'] = multi_feat.clone() # torch.Size([num_of_voxel + num_of_patch, 128])
            lidar2image_batch_dict['voxel_coords'] = torch.cat(
                [lidar2image_coords_bzyx, multiview_coords], dim=0) # torch.Size([num_of_voxel + num_of_patch, 4]) batch_id view_id y x

            lidar2image_info = self.lidar2image_input_layer(lidar2image_batch_dict)
            lidar2image_inds_list = [[lidar2image_info[f'set_voxel_inds_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
            lidar2image_masks_list = [[lidar2image_info[f'set_voxel_mask_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
            lidar2image_pos_embed_list = [[[lidar2image_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                            for i in range(self.num_shifts[s])] for b in range(self.lidar2image_pos_num)] for s in range(len(self.set_info))]

            for b in range(self.lidar2image_start, self.lidar2image_end):
                for i in range(self.num_shifts[0]):
                    multi_pos_embed_list[0][b][i] += lidar2image_pos_embed_list[0][b -
                                                                               self.lidar2image_start][i]
        if self.use_mamba_inter2:
            if self.use_mixed:
                return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, lidar2image_batch_dict['voxel_coords']
            elif self.use_multi_scale:
                return None, None, multi_pos_embed_list, torch.cat([lidar2image_coords_bzyx, multiview_coords], dim=0), lidar2image_coords_bzyx_list
            else:
                return None, None, multi_pos_embed_list, torch.cat([lidar2image_coords_bzyx, multiview_coords], dim=0)
        else:
            return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _recover_image(self, pillar_features, coords, indices):
        pillar_features = getattr(self, f'out_norm{indices}')(pillar_features)
        batch_size = coords[:, 0].max().int().item() + 1
        batch_spatial_features = pillar_features.view(
            batch_size, self.patch_size[0], self.patch_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return batch_spatial_features




class UniTRInputLayer(DSVTInputLayer):
    ''' 
    This class converts the output of vfe to unitr input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        input_image (bool): whether input modal is image.
    '''

    def __init__(self, model_cfg, accelerate=False):
        # dummy config
        model_cfg.downsample_stride = model_cfg.get('downsample_stride',[])
        model_cfg.normalize_pos = model_cfg.get('normalize_pos',False)
        super().__init__(model_cfg)

        self.input_image = self.model_cfg.get('input_image', False)
        self.key_name = 'patch' if self.input_image else 'voxel'
        # only support image input accelerate
        self.accelerate = self.input_image and accelerate
        self.process_info = None

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...

        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - ...
        '''
        if self.input_image and self.process_info is not None and (batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]):
            patch_info = dict()
            for k in (self.process_info.keys()):
                if torch.is_tensor(self.process_info[k]):
                    patch_info[k] = self.process_info[k].clone()
                else:
                    patch_info[k] = copy.deepcopy(self.process_info[k])
            # accelerate by caching pos embed as patch coords are fixed
            if not self.accelerate:
                for stage_id in range(len(self.downsample_stride)+1): # 1
                    for block_id in range(self.set_info[stage_id][1]): # 4
                        for shift_id in range(self.num_shifts[stage_id]): # 2
                            if not isinstance(self.posembed_layers[stage_id][block_id], nn.Identity):
                                patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                                    self.get_pos_embed(
                                        patch_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
                            else:
                                patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = []
            return patch_info

        key_name = self.key_name # voxel or patch
        coors = batch_dict[f'{key_name}_coords'].long() # [33792, 4]

        info = {}
        # original input voxel coors
        info[f'voxel_coors_stage0'] = coors.clone() # image_id z y x

        for stage_id in range(len(self.downsample_stride)+1): # 1
            # window partition of corrsponding stage-map
            info = self.window_partition(info, stage_id) # 有了shift前后每个voxel or patch对应的window的坐标以及在window中的坐标 4个tensor
            # generate set id of corrsponding stage-map
            info = self.get_set(info, stage_id) # 有了shift前后按照xy排序后每个set的voxel or patch的index 以及mask(重复) 4个tensor 
            for block_id in range(self.set_info[stage_id][1]): # 2  
                for shift_id in range(self.num_shifts[stage_id]): # 2
                    if not isinstance(self.posembed_layers[stage_id][block_id], nn.Identity):
                        info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                            self.get_pos_embed( # ? 这里embed为啥不引入不同view的camera信息，也没用lidar的z信息
                                info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id,  shift_id) # [num_of_voxels, 128]
                    else:
                        info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = []

        info['sparse_shape_list'] = self.sparse_shape_list # [32, 88, 1] feature的 shape lidar-camera不同

        # save process info for image input as patch coords are fixed
        if self.input_image:  
            self.process_info = {}
            for k in (info.keys()):
                if k != 'patch_feats_stage0':
                    if torch.is_tensor(info[k]):
                        self.process_info[k] = info[k].clone()
                    else:
                        self.process_info[k] = copy.deepcopy(info[k])

        return info
    
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

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id] # [30, 30, 1]
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            if self.sparse_shape[-1] == 1:
                ndim = 2
            else:
                ndim = 3
            win_x, win_y = window_shape[:2] # (30, 30)
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location) # [num_of_voxels, 2] -> [num_of_voxels, 128]

        return pos_embed



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DeformableFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, T, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1):
        super(DeformableFeatureFusion, self).__init__()
        self.offset = nn.Conv2d(in_channels * T, deformable_groups * 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels * T, out_channels * T, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=deformable_groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 如果输入和输出通道数或步长不匹配，则使用卷积调整输入形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_res = x[:, -1]
        x = x.view(B, T * C, H, W)  # 转换形状为 [B, T*C, H, W]
        offset = self.offset(x)  # 生成偏移量 [B, deformable_groups*2*kernel_size*kernel_size, H, W]
        x = self.deform_conv(x, offset)  # 应用 DeformConv2d
        x = x.view(B, T, -1, H, W).mean(dim=1)  # 转换形状为 [B, C, H, W]，并求平均
        x = self.bn(x)
        x += self.shortcut(x_res)
        x = self.relu(x)
        return x