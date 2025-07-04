import random
import warnings

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
import torch
from pcdet.datasets.augmentor.aug_utils import points_in_rbbox
import matplotlib.pyplot as plt
from PIL import Image
from tools.mambafusion_utils.vis_tools import project_lidar_to_image_torch

class VelocityAug(object):
    def __init__(self, rate=0.5, rate_vy=0.2, rate_rotation=-1, speed_range=None, thred_vy_by_vx=1.0,
                 ego_cam='CAM_FRONT'):
        # must be identical to that in tools/create_data_bevdet.py
        self.cls = ['car', 'truck', 'construction_vehicle',
                    'bus', 'trailer', 'barrier',
                    'motorcycle', 'bicycle',
                    'pedestrian', 'traffic_cone']
        self.speed_range = dict(
            car=[-10, 30, 6],
            truck=[-10, 30, 6],
            construction_vehicle=[-10, 30, 3],
            bus=[-10, 30, 3],
            trailer=[-10, 30, 3],
            barrier=[-5, 5, 3],
            motorcycle=[-2, 25, 3],
            bicycle=[-2, 15, 2],
            pedestrian=[-1, 10, 2]
        ) if speed_range is None else speed_range
        self.rate = rate
        self.thred_vy_by_vx=thred_vy_by_vx
        self.rate_vy = rate_vy
        self.rate_rotation = rate_rotation
        self.ego_cam = ego_cam

    def interpolating(self, vx, vy, delta_t, box, rot):
        delta_t_max = np.max(delta_t)
        if vy ==0 or vx == 0:
            delta_x = delta_t*vx
            delta_y = np.zeros_like(delta_x)
            rotation_interpolated = np.zeros_like(delta_x)
        else:
            theta = np.arctan2(abs(vy), abs(vx))
            rotation = 2 * theta
            radius = 0.5 * delta_t_max * np.sqrt(vx ** 2 + vy ** 2) / np.sin(theta)
            rotation_interpolated = delta_t / delta_t_max * rotation
            delta_y = radius - radius * np.cos(rotation_interpolated)
            delta_x = radius * np.sin(rotation_interpolated)
            if vy<0:
                delta_y = - delta_y
            if vx<0:
                delta_x = - delta_x
            if np.logical_xor(vx>0, vy>0):
                rotation_interpolated = -rotation_interpolated
        aug = np.zeros((delta_t.shape[0],3,3), dtype=np.float32)
        aug[:, 2, 2] = 1.
        sin = np.sin(-rotation_interpolated)
        cos = np.cos(-rotation_interpolated)
        aug[:,:2,:2] = np.stack([cos,sin,-sin,cos], axis=-1).reshape(delta_t.shape[0], 2, 2)
        aug[:,:2, 2] = np.stack([delta_x, delta_y], axis=-1)

        corner2center = np.eye(3)
        corner2center[0, 2] = -0.5 * box[3]

        instance2ego = np.eye(3)
        yaw = -box[6]
        s = np.sin(yaw)
        c = np.cos(yaw)
        instance2ego[:2,:2] = np.stack([c,s,-s,c]).reshape(2,2)
        instance2ego[:2,2] = box[:2]
        corner2ego = instance2ego @ corner2center
        corner2ego = corner2ego[None, ...]
        if not rot == 0:
            t_rot = np.eye(3)
            s_rot = np.sin(-rot)
            c_rot = np.cos(-rot)
            t_rot[:2,:2] = np.stack([c_rot, s_rot, -s_rot, c_rot]).reshape(2,2)

            instance2ego_ = np.eye(3)
            yaw_ = -box[6] - rot
            s_ = np.sin(yaw_)
            c_ = np.cos(yaw_)
            instance2ego_[:2, :2] = np.stack([c_, s_, -s_, c_]).reshape(2, 2)
            instance2ego_[:2, 2] = box[:2]
            corner2ego_ = instance2ego_ @ corner2center
            corner2ego_ = corner2ego_[None, ...]
            t_rot = instance2ego @ t_rot @ np.linalg.inv(instance2ego)
            aug = corner2ego_ @ aug @ np.linalg.inv(corner2ego_) @ t_rot[None, ...]
        else:
            aug = corner2ego @ aug @ np.linalg.inv(corner2ego)
        return aug

    def __call__(self, data_dict):
        gt_boxes = data_dict['gt_boxes'].copy()
        gt_velocity = gt_boxes[:,7:]
        gt_velocity_norm = np.sum(np.square(gt_velocity), axis=1)
        points = data_dict['points'].copy()
        point_indices = points_in_rbbox(points, gt_boxes, origin=(0.5, 0.5, 0.5))

        for bid in range(gt_boxes.shape[0]):
            cls = data_dict['gt_names'][bid]
            points_all = points[point_indices[:, bid]]
            delta_t = np.unique(points_all[:,4])
            aug_rate_cls = self.rate if isinstance(self.rate, float) else self.rate[cls]
            if points_all.shape[0]==0 or \
                    delta_t.shape[0]<3 or \
                    gt_velocity_norm[bid]>0.01 or \
                    cls not in self.speed_range or \
                    np.random.rand() > aug_rate_cls:
                continue

            # sampling speed vx,vy in instance coordinate
            vx = np.random.rand() * (self.speed_range[cls][1] -
                                     self.speed_range[cls][0]) + \
                 self.speed_range[cls][0]
            if np.random.rand() < self.rate_vy:
                max_vy = min(self.speed_range[cls][2]*2, abs(vx) * self.thred_vy_by_vx)
                vy = (np.random.rand()-0.5) * max_vy
            else:
                vy = 0.0
            vx = -vx

            # if points_all.shape[0] == 0 or cls not in self.speed_range or gt_velocity_norm[bid]>0.01 or delta_t.shape[0]<3:
            #     continue
            # vx = 10
            # vy = -2.

            rot = 0.0
            if np.random.rand() < self.rate_rotation:
                rot = (np.random.rand()-0.5) * 1.57

            aug = self.interpolating(vx, vy, delta_t, gt_boxes[bid], rot)

            # update rotation
            gt_boxes[bid, 6] += rot

            # update velocity
            delta_t_max = np.max(delta_t)
            delta_t_max_index = np.argmax(delta_t)
            center = gt_boxes[bid:bid+1, :2]
            center_aug = center @ aug[delta_t_max_index, :2, :2].T + aug[delta_t_max_index, :2, 2]
            vel = (center - center_aug) / delta_t_max
            gt_boxes[bid, 7:] = vel

            # update points
            for fid in range(delta_t.shape[0]):
                points_curr_frame_idxes = points_all[:,4] == delta_t[fid]

                points_all[points_curr_frame_idxes, :2] = \
                    points_all[points_curr_frame_idxes, :2]  @ aug[fid,:2,:2].T + aug[fid,:2, 2:3].T
            

            points[point_indices[:, bid]] = points_all


        data_dict['points'] = points
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def adjust_adj_points(self, adj_points, point_indices_adj, bid, vx, vy, rot, gt_boxes_adj, info_adj, info):
        ts_diff = info['timestamp'] / 1e6 - info_adj['timestamp'] / 1e6
        points = adj_points.tensor.numpy().copy()
        points_all_adj = points[point_indices_adj[:, bid]]
        if points_all_adj.size>0:
            delta_t_adj = np.unique(points_all_adj[:, 4]) + ts_diff
            aug = self.interpolating(vx, vy, delta_t_adj, gt_boxes_adj[bid], rot)
            for fid in range(delta_t_adj.shape[0]):
                points_curr_frame_idxes = points_all_adj[:, 4] == delta_t_adj[fid]- ts_diff
                points_all_adj[points_curr_frame_idxes, :2] = \
                    points_all_adj[points_curr_frame_idxes, :2] @ aug[fid, :2, :2].T + aug[fid, :2, 2:3].T
            points[point_indices_adj[:, bid]] = points_all_adj
        adj_points = adj_points.new_point(points)
        return adj_points
    

class PointToMultiViewDepth(object):

    def __init__(self, grid_config={
            # 'x': [-54.0, 54.0, 0.3],
            # 'y': [-54.0, 54.0, 0.3],
            # 'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 0.5],
            }, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results

class PointToMultiViewDepthFusion(PointToMultiViewDepth):
    def __call__(self, data_dict):
        # points_camego_aug = data_dict['points'].tensor[:, :3]
        points_camego_aug = torch.tensor(data_dict['points'][:, :3])
        # points_camego = points_camego_aug
        imgs_size = data_dict['camera_imgs'][0].shape
        depth_map_list = []
        for cid in range(len(data_dict['camera_imgs'])):
            # lidar2camera = data_dict['lidar2camera'][cid]
            # camera_intrinsics = data_dict['camera_intrinsics'][cid]
            # image_aug_matrix = data_dict['img_aug_matrix'][cid]
            lidar2camera = torch.tensor(data_dict['lidar2camera'][cid])
            camera_intrinsics = torch.tensor(data_dict['camera_intrinsics'][cid])
            image_aug_matrix = torch.tensor(data_dict['img_aug_matrix'][cid])
            lidar_aug_matrix = torch.tensor(data_dict['lidar_aug_matrix']).float()
            points_image, depth = project_lidar_to_image_torch(points_camego_aug, lidar2camera, camera_intrinsics, image_aug_matrix, 704, 256, lidar_aug_matrix=lidar_aug_matrix)
            points_img = torch.cat([points_image, depth.unsqueeze(1)], dim=1)
            # cam_name = data_dict['cam_names'][cid]
            # rotation_matrix = data_dict['lidar2camera'][cid][:3, :3]
            # translation_vector = data_dict['lidar2camera'][cid][:3, 3]
            # points_camera = np.dot(points_camego_aug, rotation_matrix.T) + translation_vector # 转换到相机坐标系

            # camera_intrinsics_3x3 = data_dict['camera_intrinsics'][cid][:3, :3]

            # points_camera_normalized = points_camera[:, :3] / points_camera[:, 2:3] # 进行归一化
            # # points_camera_normalized = np.concatenate([points_camera[:, :2] / points_camera[:, 2:3], points_camera[:, 2:3]], axis=1)

            # points_img = np.dot(camera_intrinsics_3x3, points_camera_normalized.T).T
            # points_img = np.concatenate([points_img[:, :2], points_camera[:, 2:3]], axis=1)
            # points_img = torch.tensor(points_img)
            # Get camera transformation matrices
            # cam2camego = torch.tensor(data_dict['lidar2camera'][cid])  # (4, 4)
            # cam2img = torch.tensor(data_dict['lidar2image'][cid])      # (4, 4)
            # cam_intrinsics = data_dict['camera_intrinsics'][cid]  # (3, 3)

            # Convert points from lidar to camera
            # points_img = points_camego.matmul(cam2img[:3, :3].T) + cam2img[:3, 3].unsqueeze(0) # [279574, 3]
            # points_img = torch.cat(
            #     [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
            #     1
            # )
            # # Transform points to image space
            # points_img = points_img.matmul(cam2img[:3, :3].T) + cam2img[:3, 3].unsqueeze(0)

            # Generate depth map
            depth_map = self.points2depthmap(points_img, imgs_size[1], imgs_size[2])
            depth_map_list.append(depth_map)

        depth_map = torch.stack(depth_map_list)
        data_dict['gt_depth'] = depth_map
        # print(points_lidar.shape)
        # print(points_lidar.shape)
        # imgs, rots, trans, intrins = data_dict['img_inputs'][:4]
        # post_rots, post_trans, bda = data_dict['img_inputs'][4:]
        # points_camego = points_camego_aug - bda[:3, 3].view(1,3)
        # points_camego = points_camego.matmul(torch.inverse(bda[:3,:3]).T)

        # depth_map_list = []
        # for cid in range(len(data_dict['cam_names'])):
        #     cam_name = data_dict['cam_names'][cid]

        #     cam2camego = np.eye(4, dtype=np.float32)
        #     cam2camego[:3, :3] = Quaternion(
        #         data_dict['curr']['cams'][cam_name]
        #         ['sensor2ego_rotation']).rotation_matrix
        #     cam2camego[:3, 3] = data_dict['curr']['cams'][cam_name][
        #         'sensor2ego_translation']
        #     cam2camego = torch.from_numpy(cam2camego)

        #     cam2img = np.eye(4, dtype=np.float32)
        #     cam2img = torch.from_numpy(cam2img)
        #     cam2img[:3, :3] = intrins[cid]

        #     camego2img = cam2img.matmul(torch.inverse(cam2camego))

        #     points_img = points_camego.matmul(
        #         camego2img[:3, :3].T) + camego2img[:3, 3].unsqueeze(0)
        #     points_img = torch.cat(
        #         [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
        #         1)
        #     points_img = points_img.matmul(
        #         post_rots[cid].T) + post_trans[cid:cid + 1, :]
        #     depth_map = self.points2depthmap(points_img, imgs.shape[2],
        #                                      imgs.shape[3])
        #     depth_map_list.append(depth_map)
        # depth_map = torch.stack(depth_map_list)
        # data_dict['gt_depth'] = depth_map
        return data_dict
def visualize_depth_and_images(depth_map_list, camera_imgs, save_path='depth_map.png'):
    num_images = len(depth_map_list)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        # 可视化深度图
        depth_map = depth_map_list[i].cpu().numpy()  # 将 tensor 转换为 numpy 数组
        depth_map = np.squeeze(depth_map)  # 删除多余的维度
        
        # 可视化对应的图像
        camera_img = camera_imgs[i]
        
        # 绘制图像和深度图
        if num_images == 1:
            axes[0].imshow(camera_img)
            axes[1].imshow(depth_map, cmap='plasma')
            axes[1].set_title(f"Depth Map {i+1}")
        else:
            axes[i, 0].imshow(camera_img)
            axes[i, 0].set_title(f"Camera Image {i+1}")
            
            im = axes[i, 1].imshow(depth_map, cmap='plasma')
            axes[i, 1].set_title(f"Depth Map {i+1}")
            
        # 添加颜色条
        fig.colorbar(im, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig(save_path)
