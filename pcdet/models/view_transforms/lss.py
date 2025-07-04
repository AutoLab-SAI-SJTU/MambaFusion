import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool
from pcdet.models.backbones_3d.local_mamba import GlobalMamba
from functools import partial
from pcdet.models.backbones_3d.lion_backbone_one_stride import LocalMamba
from easydict import EasyDict
from ...utils.spconv_utils import replace_feature, spconv
from pcdet.ops.bev_pool_v2.bev_pool import bev_pool_v2

__all__ = ["LSSTransform_Lite, LSSTransform"]

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class LSSTransform_Lite(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.use_mamba = self.model_cfg.get("USE_MAMBA", False)
        self.use_multi_block = model_cfg.get('USE_MULTI_BLOCK', False)
        self.use_pool_v2 = model_cfg.get('USE_POOL_V2', False)
        if self.use_pool_v2:
            self.grid_config = {'x': [-54.0, 54.0, 0.3], 'y': [-54.0, 54.0, 0.3], 'z': [-5.0, 3.0, 8.0], 'depth': [1.0, 60.0, 0.5]}
            self.create_grid_infos(**self.grid_config)
            self.collapse_z = model_cfg.get('collapse_z', True)
        if self.use_mamba:
            self.x_coord = None
            self.bev_size = model_cfg.get('BEV_SIZE', 360)
            self.shape_inter = [1, self.bev_size, self.bev_size]
            self.hilbert_config = {#'curve_template_path_rank10': '../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 
                                   'curve_template_path_rank9': '../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 
                                   'curve_template_path_rank8': '../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 
                                   'curve_template_path_rank7': '../ckpts/hilbert_template/curve_template_3d_rank_7.pth'}
            # self.hilbert_spatial_sis
            self.curve_template = {}
            self.template_on_device = False
            self.hilbert_spatial_size = {}
            # self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 9)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 9)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 8)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_7.pth', 7)
            self.mamba_downsample_scale = model_cfg.get('MAMBA_DOWNSAMPLE_SCALE', 1)
            self.mamba_layernorm = nn.LayerNorm(out_channel)
            self.mamba_layernorm2 = nn.LayerNorm(128)
            self.mamba_blocks = nn.ModuleList()
            for i in range(1):
                if self.use_multi_block:
                    local_block = LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=128, direction=['x', 'y'], shift=True,
                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34)
                    global_block = GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                        downsample_lvl='curve_template_rank8',
                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                        device='cuda', dtype=torch.float32)
                    self.mamba_blocks.append(local_block)
                    self.mamba_blocks.append(global_block)
                else:
                    global_block = GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                        downsample_lvl='curve_template_rank8',
                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                        device='cuda', dtype=torch.float32)
                    self.mamba_blocks.append(global_block)
            if self.mamba_downsample_scale > 1:
                assert self.mamba_downsample_scale == 2 or self.mamba_downsample_scale == 4, self.mamba_downsample_scale
                if self.mamba_downsample_scale == 2:
                    self.sub_dim = nn.Sequential(
                        nn.Conv2d(128, out_channel, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.ConvTranspose2d(out_channel, out_channel, 3, stride=self.mamba_downsample_scale, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                    )
                else:
                    self.sub_dim = nn.Sequential(
                        nn.Conv2d(128, out_channel, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                        nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU(),
                    )
            else:
                self.sub_dim = nn.Sequential(
                    nn.Conv2d(128, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                )
            self.pos_embed_inter = nn.Sequential(
                nn.Linear(9, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                )
            # out_channel = 128
            if self.mamba_downsample_scale > 1:
                if self.mamba_downsample_scale == 2:
                    self.mamba_downsample = nn.Sequential(
                        nn.Conv2d(out_channel, 128, 3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(
                            128,
                            128,
                            3,
                            stride=self.mamba_downsample_scale,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 128, 3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                    )
                else:
                    self.mamba_downsample = nn.Sequential(
                        nn.Conv2d(out_channel, 128, 3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(
                            128,
                            128,
                            3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 128, 3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(
                            128,
                            128,
                            3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(128),
                        nn.ReLU(True),
                    )

            else:
                self.mamba_downsample = nn.Sequential(
                    nn.Conv2d(out_channel, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                )
            
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE
        self.accelerate = self.model_cfg.get("ACCELERATE",False)
        if self.accelerate:
            self.cache = None

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)
        self.with_depth_from_lidar = model_cfg.get('with_depth_from_lidar', False)
        if self.with_depth_from_lidar:
            self.lidar_input_net = nn.Sequential(
                nn.Conv2d(1, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True))
            depth_out_channels = self.D + out_channel
            self.depthnet = nn.Sequential(
                nn.Conv2d(in_channel + 64, in_channel, 3, padding=1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(True),
                nn.Conv2d(in_channel, in_channel, 3, padding=1),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(True),
                nn.Conv2d(in_channel, depth_out_channels, 1))
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channel,
                    out_channel,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            # map segmentation
            if self.model_cfg.get('USE_CONV_FOR_NO_STRIDE',False):
                self.downsample = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                )
            else:
                self.downsample = nn.Identity()
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])
    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    def get_cam_feats(self, x, depth=None):
        x = x.to(torch.float)
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)
        if self.with_depth_from_lidar:
            depth_from_lidar = depth
            assert depth_from_lidar is not None
            if isinstance(depth_from_lidar, list):
                assert len(depth_from_lidar) == 1
                depth_from_lidar = depth_from_lidar[0]
            B, N = depth_from_lidar.shape[:2]
            h_img, w_img = depth_from_lidar.shape[2:]
            depth_from_lidar = depth_from_lidar.view(B * N, 1, h_img, w_img)
            depth_from_lidar = self.lidar_input_net(depth_from_lidar)
            x = torch.cat([x, depth_from_lidar], dim=1)
            x = self.depthnet(x) 
        else:
            x = self.depthnet(x) # [6, 256, 32, 88] -> [6, 246, 32, 88]
        depth = x[:, : self.D].softmax(dim=1) # [6, 118, 32, 88]
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # [6, 80, 118, 32, 88]

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2) # [1, 6, 118, 32, 88, 80]

        return x
    
    def get_cam_feats_v2(self, x):
        x = x.to(torch.float)
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x) # [6, 256, 32, 88] -> [6, 246, 32, 88]
        depth = x[:, : self.D].softmax(dim=1) # [6, 118, 32, 88]
        # x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # [6, 80, 118, 32, 88]

        # x = x.view(B, N, self.C, self.D, fH, fW)
        # x = x.permute(0, 1, 3, 4, 5, 2) # [1, 6, 118, 32, 88, 80]

        return depth, x[:, self.D : (self.D + self.C)]
    
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        lidar2img,
        img_aug_matrix
    ):
        lidar2img = lidar2img.to(torch.float)
        img_aug_matrix = img_aug_matrix.to(torch.float)

        B,N = lidar2img.shape[:2]
        D,H,W = self.frustum.shape[:3] # [118, 32, 88]
        points = self.frustum.view(1,1,D,H,W,3).repeat(B,N,1,1,1,1) # [2, 6, 118, 32, 88, 3]

        # undo post-transformation
        # B x N x D x H x W x 3
        points = torch.cat([points,torch.ones_like(points[...,-1:])],dim=-1) # [2, 6, 118, 32, 88, 4] 变成齐次坐标
        points = torch.inverse(img_aug_matrix).view(B,N,1,1,1,4,4).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
                torch.ones_like(points[:, :, :, :, :, 2:3])
            ),
            5,
        )
        points = torch.inverse(lidar2img).view(B,N,1,1,1,4,4).matmul(points).squeeze(-1)[...,:3] # [2, 6, 118, 32, 88, 3]

        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float) # [2, 6, 118, 32, 88, 3]
        x = x.to(torch.float) # [2, 6, 118, 32, 88, 80]

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C) # [2*6*118*32*88, 80]

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        if self.accelerate and self.cache is None:
            self.cache = (geom_feats,kept)

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def acc_bev_pool(self,x):
        geom_feats,kept = self.cache
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)
        x = x[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final
    
    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat
    
    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
    
    def forward(self, batch_dict):
        x = batch_dict['image_fpn'] 
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        img = x.view(int(BN/6), 6, C, H, W) # [2, 6, 256, 32, 88]
        if self.use_pool_v2:
            depth, x = self.get_cam_feats_v2(img) # [12, 118, 32, 88] [12, 80, 32, 88]
            depth = depth.view(int(BN/6), 6, 118, 32, 88) # [2, 6, 118, 32, 88]
            x = x.view(int(BN/6), 6, 80, 32, 88) # [2, 6, 80, 32, 88]
        else:
            if self.with_depth_from_lidar:
                x = self.get_cam_feats(img, batch_dict['gt_depth'])
            else:
                x = self.get_cam_feats(img) # [2, 6, 118, 32, 88, 80]
        if self.accelerate and self.cache is not None: 
            x = self.acc_bev_pool(x)
        else:
            img_aug_matrix = batch_dict['img_aug_matrix'] # [2, 6, 4, 4]
            lidar2image = batch_dict['lidar2image'] # [2, 6, 4, 4]
            if self.training  and 'lidar2image_aug' in batch_dict:
                lidar2image = batch_dict['lidar2image_aug'] # [2, 6, 4, 4]
             
            geom = self.get_geometry( # 生产视锥点 [1, 6, 118, 32, 88, 3]
                lidar2image,
                img_aug_matrix
            )
            if self.use_pool_v2:
                x = self.voxel_pooling_v2(geom, depth, x)
            else:
                x = self.bev_pool(geom, x) # [2, 80, 360, 360]

        x = self.downsample(x) # [2, 80, 360, 360]
        if self.use_mamba:
            if not self.template_on_device:
                self.template_on_device = True
                with torch.no_grad():
                    for name, _ in self.curve_template.items():
                        self.curve_template[name] = self.curve_template[name].to(x.device)
            x_down = self.mamba_downsample(x)

            x_down = x_down.permute(0,2,3,1).contiguous() # [2, 360, 360, 80]
            batch_size = x_down.size(0)
            # [2, 360 * 360, 80]
            x_down = x_down.reshape(-1, x_down.size(-1))
            # 生成[360, 360, 2]的坐标
            feature_map_size = self.bev_size // self.mamba_downsample_scale
            if self.x_coord == None:
                x_coord = torch.stack(torch.meshgrid([torch.arange(0, feature_map_size), torch.arange(0, feature_map_size)]), dim=-1).reshape(-1, 2)
                x_coord = torch.cat([torch.zeros_like(x_coord[..., :1]), torch.zeros_like(x_coord[..., :1]), x_coord], dim=-1)
                x_coord = x_coord * self.mamba_downsample_scale
                x_coord = x_coord.repeat(batch_size, 1, 1)
                for batch_idx in range(batch_size):
                    x_coord[batch_idx, :, 0] = batch_idx
                x_coord = x_coord.reshape(-1, x_coord.size(-1)).to(x.device)
                self.x_coord = x_coord
            else:
                x_coord = self.x_coord
            if x_coord.shape[0] != x_down.shape[0]:
                x_coord = x_coord[:x_down.shape[0]]
            pillar_features = batch_dict['pillar_features']
            num_pillars = pillar_features.shape[0]
            assert x_coord.shape[0] == x_down.shape[0], f"{x_coord.shape[0]} != {x_down.shape[0]}"
            new_x = torch.cat([pillar_features, x_down], dim=0)
            new_coord = torch.cat([batch_dict['voxel_coords'], x_coord], dim=0)
            # sort_index_by_batch = new_coord[:, 0].argsort()
            # sort_index_by_batch_recovers = sort_index_by_batch.argsort()
            # new_coord_sorted = new_coord[sort_index_by_batch]
            # new_x_sorted = new_x[sort_index_by_batch]
            if self.use_multi_block:
                new_x_sparse = spconv.SparseConvTensor(
                    features=new_x,
                    indices=new_coord.int(),
                    spatial_shape=self.shape_inter,
                    batch_size=batch_dict['batch_size'],
                )
                for i, block in enumerate(self.mamba_blocks):

                    if isinstance(block, LocalMamba):
                        new_x_sparse = block(new_x_sparse)
                    elif isinstance(block, GlobalMamba):
                        new_x, _ = block(new_x_sparse.features, new_coord, batch_dict['batch_size'], self.shape_inter,
                                            self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                        if i != len(self.mamba_blocks) - 1:
                            new_x_sparse = replace_feature(new_x_sparse, new_x)
                    else:
                        raise ValueError("Block type not supported")
            else:
                for block in self.mamba_blocks:
                    new_x, _ = block(new_x, new_coord, batch_dict['batch_size'], self.shape_inter,
                                            self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
            x_down = new_x[num_pillars:].reshape(batch_size, feature_map_size, feature_map_size, -1).permute(0, 3, 1, 2).contiguous()
            batch_dict['pillar_features'] = self.mamba_layernorm2(new_x[:num_pillars] + pillar_features)
            x_down = self.sub_dim(x_down)
            x = self.mamba_layernorm((x + x_down).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        batch_dict['spatial_features_img'] = x.permute(0,1,3,2).contiguous() # [2, 80, 360, 360]
        return batch_dict


class LSSTransform(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.use_mamba = self.model_cfg.get("USE_MAMBA", False)
        self.use_multi_block = model_cfg.get('USE_MULTI_BLOCK', False)
        if self.use_mamba:
            self.shape_inter = [1, 360, 360]
            self.hilbert_config = {#'curve_template_path_rank10': '../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 
                                   'curve_template_path_rank9': '../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 
                                   'curve_template_path_rank8': '../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 
                                   'curve_template_path_rank7': '../ckpts/hilbert_template/curve_template_3d_rank_7.pth'}
            # self.hilbert_spatial_sis
            self.curve_template = {}
            self.hilbert_spatial_size = {}
            # self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 9)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 9)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 8)
            self.load_template('../ckpts/hilbert_template/curve_template_3d_rank_7.pth', 7)
            self.mamba_blocks = nn.ModuleList()
            for i in range(1):
                if self.use_multi_block:
                    win_block = LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34)
                    dsb = GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                        downsample_lvl='curve_template_rank8',
                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                        device='cuda', dtype=torch.float32)
                    # dsb = DSB(DSB(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                    #     down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                    #     norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                    #     downsample_lvl='curve_template_rank9',
                    #     downsample_ori='curve_template_rank8',
                    #     down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                    #     device='cuda', dtype=torch.float32))
                    self.mamba_blocks.append(win_block)
                    self.mamba_blocks.append(dsb)
                else:
                    # dsb = DSB(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                    #         down_kernel_size=[3, 3], down_stride=[1, 1], num_down=[0, 1], 
                    #         norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                    #         downsample_lvl='curve_template_rank7',
                    #         down_resolution=False, residual_in_fp32=True, fused_add_norm=True, 
                    #         device='cuda', dtype=torch.float32)
                    dsb = GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                        downsample_lvl='curve_template_rank8',
                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                        device='cuda', dtype=torch.float32)
                    self.mamba_blocks.append(dsb)
            self.sub_dim = nn.Sequential(
                nn.Conv2d(128, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )
            self.pos_embed_inter = nn.Sequential(
                nn.Linear(9, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                )
            out_channel = 128
            
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE
        self.accelerate = self.model_cfg.get("ACCELERATE",False)
        if self.accelerate:
            self.cache = None

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False
        self.depthnet = nn.Conv2d(in_channel, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channel,
                    out_channel,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            # map segmentation
            if self.model_cfg.get('USE_CONV_FOR_NO_STRIDE',False):
                self.downsample = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True),
                )
            else:
                self.downsample = nn.Identity()
    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    def get_cam_feats(self, x):
        x = x.to(torch.float)
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x) # [6, 256, 32, 88] -> [6, 246, 32, 88]
        depth = x[:, : self.D].softmax(dim=1) # [6, 118, 32, 88]
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # [6, 80, 118, 32, 88]

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2) # [1, 6, 118, 32, 88, 80]

        return x

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        lidar2img,
        img_aug_matrix
    ):
        lidar2img = lidar2img.to(torch.float)
        img_aug_matrix = img_aug_matrix.to(torch.float)

        B,N = lidar2img.shape[:2]
        D,H,W = self.frustum.shape[:3] # [118, 32, 88]
        points = self.frustum.view(1,1,D,H,W,3).repeat(B,N,1,1,1,1) # [2, 6, 118, 32, 88, 3]

        # undo post-transformation
        # B x N x D x H x W x 3
        points = torch.cat([points,torch.ones_like(points[...,-1:])],dim=-1) # [2, 6, 118, 32, 88, 4] 变成齐次坐标
        points = torch.inverse(img_aug_matrix).view(B,N,1,1,1,4,4).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
                torch.ones_like(points[:, :, :, :, :, 2:3])
            ),
            5,
        )
        points = torch.inverse(lidar2img).view(B,N,1,1,1,4,4).matmul(points).squeeze(-1)[...,:3] # [2, 6, 118, 32, 88, 3]

        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float) # [2, 6, 118, 32, 88, 3]
        x = x.to(torch.float) # [2, 6, 118, 32, 88, 80]

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C) # [2*6*118*32*88, 80]

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        if self.accelerate and self.cache is None:
            self.cache = (geom_feats,kept)

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def acc_bev_pool(self,x):
        geom_feats,kept = self.cache
        x = x.to(torch.float)

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)
        x = x[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(self, batch_dict):
        x = batch_dict['image_fpn'] 
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        img = x.view(int(BN/6), 6, C, H, W) # [2, 6, 256, 32, 88]
        x = self.get_cam_feats(img) # [2, 6, 118, 32, 88, 80]
        if self.accelerate and self.cache is not None: 
            x = self.acc_bev_pool(x)
        else:
            img_aug_matrix = batch_dict['img_aug_matrix'] # [2, 6, 4, 4]
            lidar2image = batch_dict['lidar2image'] # [2, 6, 4, 4]
            if self.training and 'lidar2image_aug' in batch_dict:
                lidar2image = batch_dict['lidar2image_aug'] # [2, 6, 4, 4]
             
            geom = self.get_geometry( # 生产视锥点 [1, 6, 118, 32, 88, 3]
                lidar2image,
                img_aug_matrix
            )
            x = self.bev_pool(geom, x) # [2, 80, 360, 360]
        x = self.downsample(x) # [2, 80, 360, 360]
        if self.use_mamba:
            with torch.no_grad():
                for name, _ in self.curve_template.items():
                    self.curve_template[name] = self.curve_template[name].to(x.device)
            x = x.permute(0,2,3,1).contiguous() # [2, 360, 360, 80]
            batch_size = x.size(0)
            # [2, 360 * 360, 80]
            x = x.reshape(-1, x.size(-1))
            # 生成[360, 360, 2]的坐标
            x_coord = torch.stack(torch.meshgrid([torch.arange(0, 360), torch.arange(0, 360)]), dim=-1).reshape(-1, 2)
            x_coord = torch.cat([torch.zeros_like(x_coord[..., :1]), torch.zeros_like(x_coord[..., :1]), x_coord], dim=-1)
            x_coord = x_coord.repeat(batch_size, 1, 1)
            for batch_idx in range(batch_size):
                x_coord[batch_idx, :, 0] = batch_idx
            x_coord = x_coord.reshape(-1, x_coord.size(-1)).to(x.device)
            pillar_features = batch_dict['pillar_features']
            num_pillars = pillar_features.shape[0]
            new_x = torch.cat([pillar_features, x], dim=0)
            new_coord = torch.cat([batch_dict['voxel_coords'], x_coord], dim=0)
            # sort_index_by_batch = new_coord[:, 0].argsort()
            # sort_index_by_batch_recovers = sort_index_by_batch.argsort()
            # new_coord_sorted = new_coord[sort_index_by_batch]
            # new_x_sorted = new_x[sort_index_by_batch]
            if self.use_multi_block:
                new_x_sparse = spconv.SparseConvTensor(
                    features=new_x,
                    indices=new_coord.int(),
                    spatial_shape=self.shape_inter,
                    batch_size=batch_dict['batch_size'],
                )
                for i, block in enumerate(self.mamba_blocks):
                    # if i % 2 == 0:
                    #     new_x_sparse = block(new_x_sparse)
                    # else:
                    #     new_x, _ = block(new_x_sparse.features, new_coord, batch_dict['batch_size'], self.shape_inter,
                    #                         self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                    #     if i != len(self.mamba_blocks) - 1:
                    #         new_x_sparse = replace_feature(new_x_sparse, new_x)
                    if isinstance(block, LocalMamba):
                        new_x_sparse = block(new_x_sparse)
                    elif isinstance(block, GlobalMamba):
                        new_x, _ = block(new_x_sparse.features, new_coord, batch_dict['batch_size'], self.shape_inter,
                                            self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                        if i != len(self.mamba_blocks) - 1:
                            new_x_sparse = replace_feature(new_x_sparse, new_x)
                    else:
                        raise ValueError("Block type not supported")
            else:
                for block in self.mamba_blocks:
                    new_x, _ = block(new_x, new_coord, batch_dict['batch_size'], self.shape_inter,
                                            self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
            x = new_x[num_pillars:].reshape(batch_size, 360, 360, -1).permute(0, 3, 1, 2).contiguous()
            batch_dict['pillar_features'] = new_x[:num_pillars]
            x = self.sub_dim(x)
        batch_dict['spatial_features_img'] = x.permute(0,1,3,2).contiguous() # [2, 80, 360, 360]
        return batch_dict