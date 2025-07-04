import torch
import torch.nn as nn

import math
from functools import partial
from mamba_ssm.models.mixer_seq_simple import create_block
from ..model_utils.voxel_mamba_utils import get_hilbert_index_3d_mamba_lite
from pcdet.ops.win_coors.flattened_window_cuda import fused_hilbert_pos_embed as fused_hilbert_pos_embed_cuda
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import post_act_block
import torch.utils.checkpoint as cp

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



class GlobalMamba(nn.Module):
    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm,
                 down_kernel_size,
                 down_stride,
                 num_down,
                 norm_fn,
                 indice_key,
                 sparse_shape,
                 hilbert_config,
                 downsample_lvl,
                 down_resolution=True,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 downsample_ori=None,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}

        # mamba layer
        mamba_encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            **factory_kwargs,
        )

        mamba_encoder_2 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_1, mamba_encoder_2])

        # downsampling operation #
        self.conv_encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.conv_encoder.append(
                DownSp(d_model, down_kernel_size[idx], down_stride[idx], num_down[idx], norm_fn, f"{indice_key}_{idx}"))
        
        # upsampling operation #
        downsample_times = len(down_stride[1:])
        self.conv_decoder = nn.ModuleList()
        self.conv_decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]):
            if down_resolution:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                        indice_key=f'spconv_{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
            else:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='subm',
                        indice_key=f'{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
        
        self.sparse_shape = sparse_shape
        self.downsample_lvl = downsample_lvl

        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.norm = norm_cls(d_model)
        self.norm_back = norm_cls(d_model)
        self.downsample_ori = downsample_ori if downsample_ori is not None else 'curve_template_rank9'
    def forward(
        self,
        voxel_features,
        voxel_coords,
        batch_size,
        curt_spatial_shape,
        curve_template,
        hilbert_spatial_size,
        pos_embed,
        num_stage,
        debug=False,
        ):

        mamba_layer1 = self.mamba_encoder_list[0]
        mamba_layer2 = self.mamba_encoder_list[1]
        
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=curt_spatial_shape,
            batch_size=batch_size
        )

        features = []
        for conv in self.conv_encoder:
            x = conv(x)
            features.append(x)
        
        x_s1 = features[0]
        x_s2 = features[1]
        feats_s2 = features[1].features
        coords_s2 = features[1].indices
        feats_s1 = features[0].features
        coords_s1 = voxel_coords

        clvl_cruve_template_s1 = curve_template[self.downsample_ori]
        clvl_hilbert_spatial_size_s1 = hilbert_spatial_size[self.downsample_ori]
        clvl_cruve_template_s2 = curve_template[self.downsample_lvl] # 'curve_template_rank9' 
        clvl_hilbert_spatial_size_s2 = hilbert_spatial_size[self.downsample_lvl] # (1, 512, 512)
        # hilbert_s1, hilbert_s2, pos_embed_coords_s1_new, pos_embed_coords_s2_new = fused_hilbert_pos_embed_cuda(
        #     coords_s1.long(), coords_s2.long(), clvl_cruve_template_s1, clvl_cruve_template_s2, batch_size, 
        #     clvl_hilbert_spatial_size_s1[0], clvl_hilbert_spatial_size_s1[1], clvl_hilbert_spatial_size_s1[2],
        #     clvl_hilbert_spatial_size_s2[0], clvl_hilbert_spatial_size_s2[1], clvl_hilbert_spatial_size_s2[2],
        #     x_s1.spatial_shape, x_s2.spatial_shape, (num_stage, num_stage, num_stage)
        # )
        hilbert_s1, pos_embed_coords_s1= fused_hilbert_pos_embed_cuda(
            coords_s1.long(), clvl_cruve_template_s1, batch_size, 
            clvl_hilbert_spatial_size_s1[0], clvl_hilbert_spatial_size_s1[1], clvl_hilbert_spatial_size_s1[2],
            x_s1.spatial_shape, (num_stage, num_stage, num_stage)
        )
        hilbert_s2, pos_embed_coords_s2 = fused_hilbert_pos_embed_cuda(
            coords_s2.long(), clvl_cruve_template_s2, batch_size, 
            clvl_hilbert_spatial_size_s2[0], clvl_hilbert_spatial_size_s2[1], clvl_hilbert_spatial_size_s2[2],
            x_s2.spatial_shape, (num_stage, num_stage, num_stage)
        )
        # 创建 inds_curt_to_next 和 inds_next_to_curt
        inds_curt_to_next_s1 = {}
        inds_next_to_curt_s1 = {}
        inds_curt_to_next_s2 = {}
        inds_next_to_curt_s2 = {}
        index_info_s1 = {}
        index_info_s2 = {}
        for i in range(batch_size):
            batch_mask_s1 = coords_s1[:, 0] == i
            batch_mask_s2 = coords_s2[:, 0] == i

            # 对 hilbert_s1 和 hilbert_s2 进行排序
            inds_curt_to_next = torch.argsort(hilbert_s1[batch_mask_s1])
            inds_next_to_curt = torch.argsort(inds_curt_to_next)
            inds_curt_to_next_s1[i] = inds_curt_to_next
            inds_next_to_curt_s1[i] = inds_next_to_curt

            inds_curt_to_next = torch.argsort(hilbert_s2[batch_mask_s2])
            inds_next_to_curt = torch.argsort(inds_curt_to_next)
            inds_curt_to_next_s2[i] = inds_curt_to_next
            inds_next_to_curt_s2[i] = inds_next_to_curt
        index_info_s1['inds_curt_to_next'] = inds_curt_to_next_s1
        index_info_s1['inds_next_to_curt'] = inds_next_to_curt_s1
        index_info_s2['inds_curt_to_next'] = inds_curt_to_next_s2
        index_info_s2['inds_next_to_curt'] = inds_next_to_curt_s2
            
        
        pos_embed_s2 = pos_embed(pos_embed_coords_s2.float())

        inds_curt_to_next_s2 = index_info_s2['inds_curt_to_next']
        inds_next_to_curt_s2 = index_info_s2['inds_next_to_curt']
        inds_curt_to_next_s1 = index_info_s1['inds_curt_to_next']
        inds_next_to_curt_s1 = index_info_s1['inds_next_to_curt']
        new_features = []
        # Low Resolution
        out_feat_3d_s2 = torch.zeros_like(feats_s2)
        out_feat_3d_s1 = torch.zeros_like(feats_s1)

        feats_s2 = feats_s2 + pos_embed_s2

        # Borward SSMs
        for i in range(batch_size):
            b_mask_m2 = coords_s2[:, 0] == i
            feat_m2 = feats_s2[b_mask_m2][inds_curt_to_next_s2[i]][None]
            if self.training and self.use_checkpoint:
                out_feat_m2 = cp.checkpoint(mamba_layer1, feat_m2, None, use_reentrant=False)
            else:
                out_feat_m2 = mamba_layer1(feat_m2, None) # [1, 22095, 128]
            out_feat_3d_s2[b_mask_m2] = (out_feat_m2[0]).squeeze(0)[inds_next_to_curt_s2[i]]

        x_s2 = replace_feature(x_s2, self.norm(out_feat_3d_s2))


        pos_embed_s1 = pos_embed(pos_embed_coords_s1.float())

        feats_s1 = feats_s1 + pos_embed_s1
        for i in range(batch_size):
            b_mask_m1 = coords_s1[:, 0] == i
            feat_m1 = feats_s1[b_mask_m1][inds_curt_to_next_s1[i]][None]
            feat_back = feat_m1.flip(1)
            if self.training and self.use_checkpoint:
                out_feat_back = cp.checkpoint(mamba_layer2, feat_back, None, use_reentrant=False)
            else:
                out_feat_back = mamba_layer2(feat_back, None)
            out_feat_3d_s1[b_mask_m1] = (out_feat_back[0]).squeeze(0).flip(0)[inds_next_to_curt_s1[i]]

        x_s1 = replace_feature(x_s1, self.norm_back(out_feat_3d_s1))

        # new_features.append(features[0])
        new_features.append(x_s1)
        new_features.append(x_s2)

        x = x_s2

        for deconv, norm, up_x in zip(self.conv_decoder, self.conv_decoder_norm, new_features[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, x.features + up_x.features + features[0].features)
            x = replace_feature(x, norm(x.features))

        return x.features, x.indices

#####  downsampling operation  #####

class Sparse1ConvBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(Sparse1ConvBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out
    

class DownSp(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_down, norm_fn, indice_key):
        super(DownSp, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_down):
            block_list.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)
