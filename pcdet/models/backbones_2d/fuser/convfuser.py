import torch
from torch import nn

from ...vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from collections import OrderedDict
from ..base_bev_backbone import BasicBlock
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import os
import numpy as np
class ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')

        if self.merge_type == 'default':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        else:
            self.conv = nn.Sequential(
                # DepthwiseSeparableConv(in_channel, in_channel, 3, 1, 1),
                nn.Conv2d(in_channel, out_channel * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
                nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                )
        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        if self.use_merge_after:
            depths = [1]
            num_block = len(depths)
            merge_dim = 208
            self.merge_blocks = nn.ModuleList()
            # self.merge_norm = nn.ModuleList()
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            for i_layer in range(num_block):
                self.merge_blocks.append(self._make_vmamba_layer(
                    dim=merge_dim,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
        if self.use_vmamba:
            # self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            # self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)
            self.use_dw_conv = True
            depths = [1, 1, 1] # [1, 2, 2]
            num_block = len(depths)
            image_dim = 80
            point_dim = 128
            cross_dim = 128
            ssm_conv = 3
            max_channel = 1
            use_4x = False
            self.use_cross = False
            self.use_res_merge = False
            d_state = 1
            self.image_down_blocks = nn.ModuleList()
            self.image_de_blocks = nn.ModuleList()
            self.lidar_de_blocks = nn.ModuleList()
            self.lidar_down_blocks = nn.ModuleList()

            
            if self.use_res_merge:
                self.image_norm = nn.ModuleList()
                self.point_norm = nn.ModuleList()


            self.image_vmamba_blocks = nn.ModuleList()
            self.point_vmamba_blocks = nn.ModuleList()
            num_block_cross = 0

            if self.use_cross:
                depths_cross = [1, 1, 1]
                self.use_res_merge = False
                if not self.use_res_merge:
                    self.image_cross_blocks = nn.ModuleList()
                    self.point_cross_blocks = nn.ModuleList()
                
                self.image_up_blocks = nn.ModuleList()
                
                num_block_cross = len(depths_cross)
                dpr_cross = []
                for x in torch.linspace(0, 0.1, sum(depths_cross)):
                    dpr_cross.extend([x.item(), x.item()])
                self.cross_vmamba_blocks = nn.ModuleList()
                for i_layer in range(num_block_cross):
                    self.image_up_blocks.append(
                        nn.Sequential(
                            nn.Conv2d(image_dim, cross_dim, kernel_size=1),
                            nn.BatchNorm2d(cross_dim),
                            nn.ReLU(),
                            DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                        )
                    )
                    if not self.use_res_merge:
                        self.image_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, image_dim,  3, padding=1, bias=False),
                                nn.BatchNorm2d(image_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(image_dim, image_dim, 3, 1, 1),
                            )
                        )
                        self.point_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, cross_dim, 3, padding=1, bias=False),
                                nn.BatchNorm2d(cross_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                            )
                        )
                    self.cross_vmamba_blocks.append(self._make_vmamba_layer(
                        dim=cross_dim,
                        cross_dim=cross_dim,
                        drop_path = dpr_cross[sum(depths_cross[:i_layer]):sum(depths_cross[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        # =================
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='cross_noz',
                        # =================
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        cross=True,
                    ))
            if not self.use_res_merge:
                self.image_conv = nn.Sequential(
                        nn.Conv2d(image_dim * (num_block + 1), image_dim * 2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(image_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(image_dim * 2, image_dim, 3, 1, 1),
                    )
                self.lidar_conv = nn.Sequential(
                        nn.Conv2d(point_dim * (num_block + 1), point_dim *2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(point_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(point_dim * 2, point_dim, 3, 1, 1),
                    )

            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

            for i_layer in range(num_block):
                if self.use_res_merge:
                    self.image_norm.append(nn.BatchNorm2d(image_dim))
                    self.point_norm.append(nn.BatchNorm2d(point_dim))

                # if i_layer == 0 and use_4x:
                #     point_cur_layers.append(BasicBlock(point_dim, point_dim, 2, 1, True))
                if self.use_dw_conv:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(image_dim*min(i_layer + 2, max_channel), image_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(point_dim*min(i_layer + 2, max_channel), point_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                else:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                    # if i_layer == 0 and use_4x:
                    #     image_cur_layers.append(BasicBlock(image_dim, image_dim, 2, 1, True))
                    
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                self.image_down_blocks.append(nn.Sequential(*image_cur_layers))
                self.lidar_down_blocks.append(nn.Sequential(*point_cur_layers))
                
                    
                image_cur_de_layers = []
                point_cur_de_layers = []

                for j in range(i_layer + 1):
                    # if self.use_cross:
                    #     image_cur_de_layers.append(nn.ConvTranspose2d(point_dim, image_dim, kernel_size=2, stride=2, bias=False))
                    # else:
                    image_cur_de_layers.append(nn.ConvTranspose2d(image_dim*min(i_layer + 2 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    image_cur_de_layers.append(nn.BatchNorm2d(image_dim*min(i_layer + 1 - j, max_channel)))
                    image_cur_de_layers.append(nn.ReLU())
                    point_cur_de_layers.append(nn.ConvTranspose2d(point_dim*min(i_layer + 2 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    point_cur_de_layers.append(nn.BatchNorm2d(point_dim*min(i_layer + 1 - j, max_channel)))
                    point_cur_de_layers.append(nn.ReLU())
                    if self.use_dw_conv:
                        image_cur_de_layers.append(DepthwiseSeparableConv(image_dim*min(i_layer + 1 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                        point_cur_de_layers.append(DepthwiseSeparableConv(point_dim*min(i_layer + 1 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                self.image_de_blocks.append(nn.Sequential(*image_cur_de_layers))
                self.lidar_de_blocks.append(nn.Sequential(*point_cur_de_layers))
                self.image_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = image_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))

                self.point_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = point_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
    @staticmethod
    def _make_vmamba_layer(
        dim=96,
        cross_dim=0,
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
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        
        if cross_dim != 0:
            blocks1 = []
            blocks2 = []
        else:
            blocks = []
        for d in range(depth):
            if cross_dim != 0:
                blocks1.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=cross_dim,
                    drop_path=drop_path[d],
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
                blocks2.append(VSSBlock(
                    hidden_dim=cross_dim, 
                    cross_dim=dim,
                    drop_path=drop_path[d],
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
            else:
                blocks.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=0,
                    drop_path=drop_path[d],
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
        if not cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks1),
                blocks2=nn.Sequential(*blocks2),
            ))
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        lidar_bev = batch_dict['spatial_features']
        if self.use_vmamba:
            if self.use_checkpoint:
                cat_bev = checkpoint.checkpoint(self.mamba_forward, img_bev, lidar_bev)
            else:
                cat_bev = self.mamba_forward(img_bev, lidar_bev)
        else:
            cat_bev = torch.cat([img_bev, lidar_bev], dim=1)
        if self.use_merge_after:
            for block in self.merge_blocks:
                cat_bev = block(cat_bev)
        mm_bev = self.conv(cat_bev) # [2, 128, 360, 360]

        batch_dict['spatial_features'] = mm_bev
        return batch_dict

    def mamba_forward(self, img_bev, lidar_bev):
        ups_img = []
        ups_img.append(img_bev)
        ups_lidar = []
        ups_lidar.append(lidar_bev)
        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks, self.point_vmamba_blocks)):
            img_bev = self.image_down_blocks[i](img_bev) # [2, 80, 90, 90]
            img_bev = block_img(img_bev)
            lidar_bev = self.lidar_down_blocks[i](lidar_bev)
            lidar_bev = block_lidar(lidar_bev)
            if self.use_cross:
                img_bev = self.image_up_blocks[i](img_bev) # [batch_size, 128, 180, 180]
                img_bev_cross = self.cross_vmamba_blocks[i].blocks1((img_bev, lidar_bev)) # [batch_size, 128, 180, 180]
                lidar_bev_cross = self.cross_vmamba_blocks[i].blocks2((lidar_bev, img_bev))
                if not self.use_res_merge:
                    img_bev = self.image_cross_blocks[i](torch.cat([img_bev, lidar_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                    lidar_bev = self.point_cross_blocks[i](torch.cat([lidar_bev, img_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                else:
                    img_bev = img_bev_cross
                    lidar_bev = lidar_bev_cross
            if self.use_res_merge:
                img_bev = self.image_norm[i](img_bev + self.image_de_blocks[i](img_bev))
                lidar_bev = self.point_norm[i](lidar_bev + self.lidar_de_blocks[i](lidar_bev))
            else:
                ups_img.append(self.image_de_blocks[i](img_bev))
                ups_lidar.append(self.lidar_de_blocks[i](lidar_bev))
        if self.use_res_merge:
            merge_img = img_bev
            merge_lidar = lidar_bev
        else:
            merge_img = self.image_conv(torch.cat(ups_img, dim=1)) # [2, 80, 360, 360]
            merge_lidar = self.lidar_conv(torch.cat(ups_lidar, dim=1)) # [2, 128, 360, 360]
        cat_bev = torch.cat([merge_img,merge_lidar],dim=1)

        return cat_bev

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