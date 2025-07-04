import torch
import torch.nn as nn
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import three_nn
from pcdet.ops.win_coors.flattened_window_cuda import map_points as map_points_cuda
from pcdet.ops.win_coors.flattened_window_cuda import map_points_v2 as map_points_v2_cuda
# import Tuple
from typing import Tuple
# @torch.jit.script
def get_points(pc_range, sample_num, space_shape, coords=None):
    '''Generate points in specified range or voxels

    Args:
        pc_range (list(int)): point cloud range, (x1,y1,z1,x2,y2,z2)
        sample_num (int): sample point number in a voxel
        space_shape (list(int)): voxel grid shape, (w,h,d)
        coords (tensor): generate points in specified voxels, (N,3)

    Returns:
        points (tensor): generated points, (N,sample_num,3)
    '''
    sx, sy, sz = space_shape # [360, 360, 1]
    x1, y1, z1, x2, y2, z2 = pc_range # [-54.0, -54.0, -10.0, 54.0, 54.0, 10.0]
    if coords is None:
        coord_x = torch.linspace( # torch.Size([1, 360, 360, 1])
            0, sx-1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
        coord_y = torch.linspace( # torch.Size([1, 360, 360, 1])
            0, sy-1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
        coord_z = torch.linspace(   # torch.Size([1, 360, 360, 1])
            0, sz-1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
        coords = torch.stack((coord_x, coord_y, coord_z), -1).view(-1, 3) # torch.Size([129600, 3]) 360*360*1=129600
    points = coords.clone().float()
    points[..., 0] = ((points[..., 0]+0.5)/sx)*(x2-x1) + x1 # 在x轴上均匀分布，然后映射到pc_range的范围内 +0.5是为了保证在x1和x2之间
    points[..., 1] = ((points[..., 1]+0.5)/sy)*(y2-y1) + y1
    points[..., 2] = ((points[..., 2]+0.5)/sz)*(z2-z1) + z1

    if sample_num == 1:
        points = points.unsqueeze(1)
    else:
        points = points.unsqueeze(1).repeat(1, sample_num, 1) # torch.Size([129600, 20, 3])
        points[..., 2] = torch.linspace(z1, z2, sample_num).unsqueeze(0) # 在z轴上均匀分布，然后映射到pc_range的范围内(-10 -> 10)
    return points

# @torch.jit.script
def map_points(points, lidar2image, image_aug_matrix, batch_size: int, image_shape: Tuple[int, int], expand_scale: float = 0.0):
    '''Map 3D points to image space.

    Args:
        points (tensor): Grid points in 3D space, shape (grid num, sample num,4).
        lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
        image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
        batch_size (int): Sample number in a batch.
        image_shape (tuple(int)): Image shape, (height, width).

    Returns:
        points (tensor): 3d coordinates of points mapped in image space. shape (B,N,num,k,4)
        points_2d (tensor): 2d coordinates of points mapped in image space. (B,N,num,k,2)
        map_mask (tensor): Number of points per view (batch size x view num). (B,N,num,k,1)
    '''
    points = points.to(torch.float32) # torch.Size([32338, sample_num, 3])
    lidar2image = lidar2image.to(torch.float32) # torch.Size([1, 6, 4, 4])
    image_aug_matrix = image_aug_matrix.to(torch.float32) # torch.Size([1, 6, 4, 4])
 
    num_view = lidar2image.shape[1] # 6
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1) # torch.Size([32338, sample_num, 4]) 在最后一维增加一个1，转换为齐次坐标
    # map points from lidar to (aug) image space
    points = points.unsqueeze(0).unsqueeze(0).repeat(
        batch_size, num_view, 1, 1, 1).unsqueeze(-1) # torch.Size([1, 6, 32338, sample_num, 4, 1])
    grid_num, sample_num = points.shape[2:4] # 129600, sample_num
    lidar2image = lidar2image.view(batch_size, num_view, 1, 1, 4, 4).repeat( # 生成每一个点对应的变换矩阵 torch.Size([2, 6, 1, 1, 4, 4])
        1, 1, grid_num, sample_num, 1, 1) # torch.Size([2, 6, 32338, sample_num, 4, 4])
    image_aug_matrix = image_aug_matrix.view(
        batch_size, num_view, 1, 1, 4, 4).repeat(1, 1, grid_num, sample_num, 1, 1) # torch.Size([2, 6, 129600, sample_num, 4, 4])
    points_2d = torch.matmul(lidar2image, points).squeeze(-1) # 将这些点从激光雷达坐标系映射到相机坐标系 torch.Size([2, 6, 32338, sample_num, 4, 1]) -> torch.Size([2, 6, 32338, sample_num, 4])

    # recover image augmentation
    eps = 1e-5
    map_mask = (points_2d[..., 2:3] > eps) # 保留投影在相机坐标系下z轴大于0的点 torch.Size([2, 6, 32338, sample_num, 1])
    points_2d[..., 0:2] = points_2d[..., 0:2] / torch.maximum(
        points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3]) * eps) # torch.Size([2, 6, 32338, sample_num, 2]) 除以z轴，得到图像坐标系下的坐标
    points_2d[..., 2] = torch.ones_like(points_2d[..., 2]) # torch.Size([2, 6, 32338, sample_num, 1]) z轴设为1 得到齐次的图像坐标系下的坐标
    points_2d = torch.matmul( # image_aug_matrix包括了图像增强的变换矩阵，如旋转、平移、缩放等
        image_aug_matrix, points_2d.unsqueeze(-1)).squeeze(-1)[..., 0:2] # torch.Size([2, 6, 32338, sample_num, 2]) 乘以图像增强的变换矩阵
    points_2d[..., 0] /= image_shape[1] # torch.Size([2, 6, 32338, sample_num, 1]) 归一化到图像坐标系下的坐标
    points_2d[..., 1] /= image_shape[0] # torch.Size([2, 6, 32338, sample_num, 1]) 归一化到图像坐标系下的坐标

    # mask points out of range
    map_mask = (map_mask & (points_2d[..., 1:2] > 0.0 - expand_scale) 
                & (points_2d[..., 1:2] < 1.0 + expand_scale)
                & (points_2d[..., 0:1] < 1.0 + expand_scale)
                & (points_2d[..., 0:1] > 0.0 - expand_scale)) # torch.Size([2, 6, 32338, sample_num, 1]) 保留在图像范围内的点
    map_mask = torch.nan_to_num(map_mask).squeeze(-1)

    return points.squeeze(-1), points_2d, map_mask


class MapImage2Lidar(nn.Module):
    '''Map image patch to lidar space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range # [-54.0, -54.0, -10.0, 54.0, 54.0, 10.0]
        self.voxel_size = model_cfg.voxel_size # [0.3, 0.3, 20.0] 
        self.sample_num = model_cfg.sample_num  # 20 sample num in a voxel
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)] # [360, 360, 1]

        self.points = get_points(
            self.pc_range, self.sample_num, self.space_shape).cuda()
        self.accelerate = accelerate
        if self.accelerate:
            self.cache = None
        self.use_map = use_map

    def forward(self, batch_dict):
        '''Get the coordinates of image patch in 3D space.

        Returns:
            image2lidar_coords_zyx (tensor): The coordinates of image features 
            (batch size x view num) in 3D space.
            nearest_dist (tensor): The distance between each image feature 
            and the nearest mapped 3d grid point in image space.
        '''

        # accelerate by caching when the mapping relationship changes little
        if self.accelerate and self.cache is not None:
            image2lidar_coords_zyx, nearest_dist = self.cache
            return image2lidar_coords_zyx, nearest_dist
        img = batch_dict['camera_imgs'] # [batch, 6, 3, 256, 704]
        batch_size, num_view, _, h, w = img.shape
        points = self.points.clone() # torch.Size([129600, 20, 3]) 根据给定的点云范围、采样数量和空间形状，生成一个均匀分布的点集，用于映射图像块到激光雷达空间
        lidar2image = batch_dict['lidar2image'] # torch.Size([2, 6, 4, 4]) 从激光雷达空间到图像空间的变换矩阵
        image_aug_matrix = batch_dict['img_aug_matrix'] # torch.Size([2, 6, 4, 4]) 图像增强的变换矩阵

        with torch.no_grad():
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4]) 从激光雷达空间到图像空间的变换矩阵
            # get mapping points in image space
            points_3d, points_2d, map_mask = map_points(
                points, lidar2image, image_aug_matrix, batch_size, (h, w)) # torch.Size([2, 6, 129600, 20, 4]) torch.Size([2, 6, 129600, 20, 2]) torch.Size([2, 6, 129600, 20, 1])
            mapped_points_2d = points_2d[map_mask] # torch.Size([3779754, 2])
            mapped_points_3d = points_3d[map_mask] # torch.Size([3779754, 4])
            mapped_view_cnts = map_mask.view(
                batch_size, num_view, -1).sum(-1).view(-1).int() # torch.Size([batch * 6]) 每个batch中每个视角的点的数量
            mapped_points = torch.cat(
                [mapped_points_2d, torch.zeros_like(mapped_points_2d[:, :1])], dim=-1) # torch.Size([3779754, 3]) 在z轴上增加一个0
            mapped_coords_3d = mapped_points_3d[:, :3]

            # shape (H*W,2), [[x1,y1],...]
            patch_coords_perimage = batch_dict['patch_coords'][batch_dict['patch_coords'][:, 0] == 0, 2:].clone(
            ).float() # [33792, 4] -> [2816, 2] 从patch_coords中取出每个图像块的坐标
            patch_coords_perimage[:, 0] = (
                patch_coords_perimage[:, 0] + 0.5) / batch_dict['hw_shape'][1] # 归一化到图像坐标系下的坐标
            patch_coords_perimage[:, 1] = (
                patch_coords_perimage[:, 1] + 0.5) / batch_dict['hw_shape'][0] # 归一化到图像坐标系下的坐标1

            # get image patch coords
            patch_points = patch_coords_perimage.unsqueeze(
                0).repeat(batch_size * num_view , 1, 1).view(-1, 2) # torch.Size([33792, 2])
            patch_points = torch.cat( # torch.Size([33792, 3])
                [patch_points, torch.zeros_like(patch_points[:, :1])], dim=-1)
            patch_view_cnts = (torch.ones_like(
                mapped_view_cnts) * (batch_dict['hw_shape'][0] * batch_dict['hw_shape'][1])).int() # torch.Size([batch * 6]) 每个batch中每个视角的点的数量

            # find the nearest 3 mapping points and keep the closest
            _, idx = three_nn(patch_points.to(torch.float32), patch_view_cnts, mapped_points.to( # 每个图像块中的点到最近的三个映射点的索引
                torch.float32), mapped_view_cnts) # torch.Size([33792, 3]) torch.Size([33792, 3])
            idx = idx[:, :1].repeat(1, 3).long() #? torch.Size([33792, 3]) 为什么只保留最近的一个点的索引
            # take 3d coords of the nearest mapped point of each image patch as its 3d coords
            image2lidar_coords_xyz = torch.gather(mapped_coords_3d, 0, idx) # torch.Size([33792, 3])

            # calculate distance between each image patch and the nearest mapping point in image space
            neighbor_2d = torch.gather(mapped_points, 0, idx) # torch.Size([33792, 3])
            nearest_dist = (patch_points[:, :2]-neighbor_2d[:, :2]).abs() # torch.Size([33792, 2]) 计算图像块中心点到最近的映射点的距离
            nearest_dist[:, 0] *= batch_dict['hw_shape'][1] # torch.Size([33792, 2])
            nearest_dist[:, 1] *= batch_dict['hw_shape'][0] # torch.Size([33792, 2])

            # 3d coords -> voxel grids
            image2lidar_coords_xyz[..., 0] = (image2lidar_coords_xyz[..., 0] - self.pc_range[0]) / (
                self.pc_range[3]-self.pc_range[0]) * self.space_shape[0] - 0.5 # 归一化到激光雷达空间下的坐标
            image2lidar_coords_xyz[..., 1] = (image2lidar_coords_xyz[..., 1] - self.pc_range[1]) / (
                self.pc_range[4]-self.pc_range[1]) * self.space_shape[1] - 0.5 # 归一化到激光雷达空间下的坐标
            image2lidar_coords_xyz[..., 2] = 0.

            image2lidar_coords_xyz[..., 0] = torch.clamp(
                image2lidar_coords_xyz[..., 0], min=0, max=self.space_shape[0]-1) # 限制在激光雷达空间的范围内 0 -> 360
            image2lidar_coords_xyz[..., 1] = torch.clamp(
                image2lidar_coords_xyz[..., 1], min=0, max=self.space_shape[1]-1) # 限制在激光雷达空间的范围内 0 -> 360

            # reorder to z,y,x
            image2lidar_coords_zyx = image2lidar_coords_xyz[:, [2, 1, 0]] # torch.Size([33792, 3])
        if self.accelerate:
            self.cache = (image2lidar_coords_zyx, nearest_dist)
        return image2lidar_coords_zyx, nearest_dist # torch.Size([33792, 3]) torch.Size([33792, 2])


class MapLidar2Image(nn.Module):
    '''Map Lidar points to image space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False, use_denoise=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]
        self.accelerate = accelerate
        if self.accelerate:
            raise NotImplementedError
            self.full_lidar2image_coors_zyx = None
            # only support one point in a voxel
            self.points = get_points(
                self.pc_range, self.sample_num, self.space_shape).cuda()
        self.use_map = use_map
        self.use_denoise = use_denoise

    def pre_compute(self, batch_dict):
        '''Precalculate the coords of all voxels mapped on the image'''
        image = batch_dict['camera_imgs']
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        image_shape = image.shape[-2:]
        assert image.shape[0] == 1, 'batch size should be 1 in pre compute'
        batch_idx = torch.zeros(
            self.space_shape[0]*self.space_shape[1], device=image.device)
        with torch.no_grad():
            # get reference points, only in voxels.
            points = self.points.clone()
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, image_aug_matrix, batch_idx, image_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            self.full_lidar2image_coors_zyx = lidar2image_coords_xyz[:, [
                2, 0, 1]]

    def map_lidar2image(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape, coords_ori):
        '''Map Lidar points to image space.

        Args:
            points (tensor): batch lidar points shape (voxel num, sample num,4).
            lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
            image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
            batch_idx (tensor): batch id for all points in batch
            image_shape (Tuple(int, int)): Image shape, (height, width).

        Returns:
            batch_hit_points: 2d coordinates of lidar points mapped in image space. 
        '''
        num_view = lidar2image.shape[1] # 6
        batch_size = (batch_idx[-1] + 1).int() 
        batch_hit_points = []
        for b in range(batch_size):

            hit_points = map_points_v2_cuda(points[batch_idx == b],  lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape[0], image_shape[1], 0)[0].squeeze(1)
            batch_hit_points.append(hit_points)
        batch_hit_points = torch.cat(batch_hit_points, dim=0)
        return batch_hit_points

    def forward(self, batch_dict, use_multi_scale=False, space_shape=None):
        '''Get the coordinates of lidar poins in image space.

        Returns:
            lidar2image_coords_zyx (tensor): The coordinates of lidar points in 3D space.
        '''
        if self.accelerate:
            raise NotImplementedError
            if self.full_lidar2image_coors_zyx is None:
                self.pre_compute(batch_dict)
            # accelerate by index table
            coords_xyz = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone()
            unique_index = coords_xyz[:, 1] * \
                self.space_shape[1] + coords_xyz[:, 2]
            lidar2image_coords_zyx = self.full_lidar2image_coors_zyx[unique_index.long(
            )]
            return lidar2image_coords_zyx
        img = batch_dict['camera_imgs'] # torch.Size([batch_size, 6, 3, 256, 704])
        coords = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone() # torch.Size([num_of_voxel(39590), 4]) batch_idx, x, y, z
        if 'ori_coords_height' in batch_dict:
        #    ori_coords_height = (batch_dict['ori_coords_height'] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           ori_coords_height = batch_dict['ori_coords_height'].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           space_shape = [360, 360, 32]
           coords = torch.cat([coords[:, :-1], ori_coords_height], dim=1)
        lidar2image = batch_dict['lidar2image'] # torch.Size([2, 6, 4, 4])
        img_aug_matrix = batch_dict['img_aug_matrix'] # torch.Size([2, 6, 4, 4])
        hw_shape = batch_dict['hw_shape'] # (32, 88) image shape / 8

        img_shape = img.shape[-2:] # (256, 704)
        batch_idx = coords[:, 0] # torch.Size([num_of_voxel])
        with torch.no_grad():
            # get reference points, only in voxels.
            space_shape = self.space_shape if space_shape is None else space_shape
            points = get_points(self.pc_range, self.sample_num,
                                space_shape, coords[:, 1:]) # 获取每个体素中的采样点 torch.Size([num_of_voxel, 1, 3])
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4])
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, img_aug_matrix, batch_idx, img_shape, coords) # torch.Size([num_of_voxel, 3])

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1] # 归一化到图像坐标系下的坐标
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]  # 归一化到图像坐标系下的坐标
            lidar2image_coords_zyx = lidar2image_coords_xyz[:, [2, 0, 1]] # torch.Size([num_of_voxel, 3]) view_idx, x, y
        if use_multi_scale: 
            use_multi_name_list = ['x_conv3'] # ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']
            if 'ori_coords_height' in batch_dict:
                use_multi_name_coords_list = ['ori_coords_height_coords3'] # ['ori_coords_height_coords1', 'ori_coords_height_coords2', 'ori_coords_height_coords3', 'ori_coords_height_coords4']
            lidar2image_coords_zyx_list = []
            for i, name in enumerate(use_multi_name_list):
                indices = batch_dict['multi_scale_3d_features'][name].indices[:, [0, 3, 2, 1]].clone()
                if 'ori_coords_height' in batch_dict:
                    # ori_coords_height_tmp = (batch_dict[use_multi_name_coords_list[i]] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    ori_coords_height_tmp = batch_dict[use_multi_name_coords_list[i]].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    indices = torch.cat([indices[:, :-1], ori_coords_height_tmp], dim=1)
                    space_shape = [360, 360, 32]
                else:
                    space_shape = batch_dict['multi_scale_3d_features'][name].spatial_shape[::-1]
                with torch.no_grad():
                    points = get_points(self.pc_range, self.sample_num, space_shape, indices[:, 1:])
                    lidar2image_coords_xyz = self.map_lidar2image(
                        points, lidar2image, img_aug_matrix, indices[:, 0], img_shape, indices)
                    lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
                    lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
                    lidar2image_coords_zyx_tmp = lidar2image_coords_xyz[:, [2, 0, 1]]
                    lidar2image_coords_bzyx = torch.cat([indices[:, 0:1], lidar2image_coords_zyx_tmp], dim=1)
                lidar2image_coords_zyx_list.append(lidar2image_coords_bzyx)
            return lidar2image_coords_zyx, lidar2image_coords_zyx_list
        return lidar2image_coords_zyx, None


class MapLidar2Image2(nn.Module):
    '''Map Lidar points to image space'''

    def __init__(self, model_cfg, accelerate=False, use_map=False, use_denoise=False) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = model_cfg.point_cloud_range
        self.voxel_size = model_cfg.voxel_size
        self.sample_num = model_cfg.sample_num
        self.space_shape = [
            int((self.pc_range[i+3]-self.pc_range[i])/self.voxel_size[i]) for i in range(3)]
        self.accelerate = accelerate
        if self.accelerate:
            raise NotImplementedError
            self.full_lidar2image_coors_zyx = None
            # only support one point in a voxel
            self.points = get_points(
                self.pc_range, self.sample_num, self.space_shape).cuda()
        self.use_map = use_map
        self.use_denoise = use_denoise

    def pre_compute(self, batch_dict):
        '''Precalculate the coords of all voxels mapped on the image'''
        image = batch_dict['camera_imgs']
        lidar2image = batch_dict['lidar2image']
        image_aug_matrix = batch_dict['img_aug_matrix']
        hw_shape = batch_dict['hw_shape']

        image_shape = image.shape[-2:]
        assert image.shape[0] == 1, 'batch size should be 1 in pre compute'
        batch_idx = torch.zeros(
            self.space_shape[0]*self.space_shape[1], device=image.device)
        with torch.no_grad():
            # get reference points, only in voxels.
            points = self.points.clone()
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug']
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, image_aug_matrix, batch_idx, image_shape)

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
            self.full_lidar2image_coors_zyx = lidar2image_coords_xyz[:, [
                2, 0, 1]]

    def map_lidar2image(self, points, lidar2image, image_aug_matrix, batch_idx, image_shape, coords_ori):
        '''Map Lidar points to image space.

        Args:
            points (tensor): batch lidar points shape (voxel num, sample num,4).
            lidar2image (tensor): Transformation matrix from lidar to image space, shape (B, N, 4, 4).
            image_aug_matrix (tensor): Transformation of image augmentation, shape (B, N, 4, 4).
            batch_idx (tensor): batch id for all points in batch
            image_shape (Tuple(int, int)): Image shape, (height, width).

        Returns:
            batch_hit_points: 2d coordinates of lidar points mapped in image space. 
        '''
        num_view = lidar2image.shape[1] # 6
        batch_size = (batch_idx[-1] + 1).int() 
        batch_hit_points = []
        for b in range(batch_size):
            if self.use_denoise:
                _, points_2d, map_mask = map_points( # 将点从激光雷达坐标系映射到图像坐标系 torch.Size([1, 6, num_of_voxel, 1, 2]) torch.Size([1, 6, num_of_voxel, 1])
                    points[batch_idx == b], lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape, expand_scale=0.2)
            else:
                _, points_2d, map_mask = map_points( # 将点从激光雷达坐标系映射到图像坐标系 torch.Size([1, 6, num_of_voxel, 1, 2]) torch.Size([1, 6, num_of_voxel, 1])
                    points[batch_idx == b], lidar2image[b:b+1], image_aug_matrix[b:b+1], 1, image_shape)
            points_2d = points_2d.squeeze(3) # torch.Size([1, 6, num_of_voxel, 2])
            # set point not hit image as hit 0
            map_mask = map_mask.squeeze(3).permute(0, 2, 1).view(-1, num_view) # torch.Size([num_of_voxel, 6])
            hit_mask = map_mask.any(dim=-1) # torch.Size([num_of_voxel]) 有一个视角命中就算命中

            map_mask[~hit_mask, 0] = True # torch.Size([num_of_voxel, 6]) 不命中的都放视角0
            # get hit view id
            hit_view_ids = torch.nonzero(map_mask) # torch.Size([num_of_hitvoxel, 2]) 第一列是batch_idx，第二列是view_idx，由于一个voxel可能被多个视角命中，所以会有多行
            # select first view if hit multi view
            hit_poins_id = hit_view_ids[:, 0] # torch.Size([num_of_hitvoxel]) 
            shift_hit_points_id = torch.roll(hit_poins_id, 1) # torch.Size([num_of_hitvoxel])
            shift_hit_points_id[0] = -1
            first_mask = (hit_poins_id - shift_hit_points_id) > 0 # torch.Size([num_of_hitvoxel]) 保留第一个命中的视角
            unique_hit_view_ids = hit_view_ids[first_mask, 1:] # torch.Size([num_of_hitvoxel, 1]) 保留第一个命中的视角的voxel索引
            num = points_2d.shape[2] # num_of_voxel
            assert len(unique_hit_view_ids) == num, 'some points not hit view!'
            # get coords in hit view
            points_2d = points_2d.permute(0, 2, 1, 3).flatten(0, 1) # torch.Size([num_of_voxel, 6, 2]) 
            hit_points_2d = points_2d[range( # torch.Size([num_of_voxel, 2]) 保留命中的视角的点
                num), unique_hit_view_ids.squeeze()]
            # if self.use_denoise and hit_mask.sum() < num:
            #     coords_ori_current_batch = coords_ori[batch_idx == b]
            #     hit_points_2d[~hit_mask] = (coords_ori_current_batch[~hit_mask, :2] - 180) / coords_ori_current_batch.new_tensor([image_shape[1], image_shape[0]])
            #     unique_hit_view_ids[~hit_mask] = 6
            # clamp value range and adjust to postive for set partition
            hit_points_2d = torch.clamp(hit_points_2d, -1, 2) + 1 # torch.Size([num_of_voxel, 2])
            hit_points = torch.cat([hit_points_2d, unique_hit_view_ids], -1) # torch.Size([num_of_voxel, 3])
            batch_hit_points.append(hit_points)
        batch_hit_points = torch.cat(batch_hit_points, dim=0)
        return batch_hit_points

    def forward(self, batch_dict, use_multi_scale=False, space_shape=None):
        '''Get the coordinates of lidar poins in image space.

        Returns:
            lidar2image_coords_zyx (tensor): The coordinates of lidar points in 3D space.
        '''
        if self.accelerate:
            raise NotImplementedError
            if self.full_lidar2image_coors_zyx is None:
                self.pre_compute(batch_dict)
            # accelerate by index table
            coords_xyz = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone()
            unique_index = coords_xyz[:, 1] * \
                self.space_shape[1] + coords_xyz[:, 2]
            lidar2image_coords_zyx = self.full_lidar2image_coors_zyx[unique_index.long(
            )]
            return lidar2image_coords_zyx
        img = batch_dict['camera_imgs'] # torch.Size([batch_size, 6, 3, 256, 704])
        coords = batch_dict['voxel_coords'][:, [0, 3, 2, 1]].clone() # torch.Size([num_of_voxel(39590), 4]) batch_idx, x, y, z
        if 'ori_coords_height' in batch_dict:
        #    ori_coords_height = (batch_dict['ori_coords_height'] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           ori_coords_height = batch_dict['ori_coords_height'].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
           space_shape = [360, 360, 32]
           coords = torch.cat([coords[:, :-1], ori_coords_height], dim=1)
        lidar2image = batch_dict['lidar2image'] # torch.Size([2, 6, 4, 4])
        img_aug_matrix = batch_dict['img_aug_matrix'] # torch.Size([2, 6, 4, 4])
        hw_shape = batch_dict['hw_shape'] # (32, 88) image shape / 8

        img_shape = img.shape[-2:] # (256, 704)
        batch_idx = coords[:, 0] # torch.Size([num_of_voxel])
        with torch.no_grad():
            # get reference points, only in voxels.
            space_shape = self.space_shape if space_shape is None else space_shape
            points = get_points(self.pc_range, self.sample_num,
                                space_shape, coords[:, 1:]) # 获取每个体素中的采样点 torch.Size([num_of_voxel, 1, 3])
            if self.training and 'lidar2image_aug' in batch_dict and not self.use_map:
                lidar2image = batch_dict['lidar2image_aug'] # torch.Size([2, 6, 4, 4])
            # get mapping points in image space
            lidar2image_coords_xyz = self.map_lidar2image(
                points, lidar2image, img_aug_matrix, batch_idx, img_shape, coords) # torch.Size([num_of_voxel, 3])

            lidar2image_coords_xyz[:,
                                   0] = lidar2image_coords_xyz[:, 0] * hw_shape[1] # 归一化到图像坐标系下的坐标
            lidar2image_coords_xyz[:,
                                   1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]  # 归一化到图像坐标系下的坐标
            lidar2image_coords_zyx = lidar2image_coords_xyz[:, [2, 0, 1]] # torch.Size([num_of_voxel, 3]) view_idx, x, y
        if use_multi_scale: 
            use_multi_name_list = ['x_conv3'] # ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']
            if 'ori_coords_height' in batch_dict:
                use_multi_name_coords_list = ['ori_coords_height_coords3'] # ['ori_coords_height_coords1', 'ori_coords_height_coords2', 'ori_coords_height_coords3', 'ori_coords_height_coords4']
            lidar2image_coords_zyx_list = []
            for i, name in enumerate(use_multi_name_list):
                indices = batch_dict['multi_scale_3d_features'][name].indices[:, [0, 3, 2, 1]].clone()
                if 'ori_coords_height' in batch_dict:
                    # ori_coords_height_tmp = (batch_dict[use_multi_name_coords_list[i]] + 0.5).to(torch.int32).clamp(min=0, max=31).reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    ori_coords_height_tmp = batch_dict[use_multi_name_coords_list[i]].reshape(-1, 1) # torch.Size([num_of_voxel, 1]) 0 -> 31
                    indices = torch.cat([indices[:, :-1], ori_coords_height_tmp], dim=1)
                    space_shape = [360, 360, 32]
                else:
                    space_shape = batch_dict['multi_scale_3d_features'][name].spatial_shape[::-1]
                with torch.no_grad():
                    points = get_points(self.pc_range, self.sample_num, space_shape, indices[:, 1:])
                    lidar2image_coords_xyz = self.map_lidar2image(
                        points, lidar2image, img_aug_matrix, indices[:, 0], img_shape, indices)
                    lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
                    lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
                    lidar2image_coords_zyx_tmp = lidar2image_coords_xyz[:, [2, 0, 1]]
                    lidar2image_coords_bzyx = torch.cat([indices[:, 0:1], lidar2image_coords_zyx_tmp], dim=1)
                lidar2image_coords_zyx_list.append(lidar2image_coords_bzyx)
            return lidar2image_coords_zyx, lidar2image_coords_zyx_list
        return lidar2image_coords_zyx, None