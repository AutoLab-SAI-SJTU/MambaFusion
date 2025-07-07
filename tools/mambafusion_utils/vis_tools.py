
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch import Tensor
import torch

save_path = 'tools/my_utils/vis_res/'

def draw_pic_multi_view(images, img_type='Image', save_path=None):
    if len(images) != 6:
        raise ValueError("需要正好6个图像")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
        # ax.set_title(f'{img_type} {i+1}')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
def rotate_bboxes_3d(bboxes_3d, rotation_matrix):
    """
    旋转3D边界框。
    :param bboxes_3d: [N, 7] 形式的边界框，每个边界框格式为 (x, y, z, l, w, h, yaw)
    :param rotation_matrix: [3, 3] 形式的旋转矩阵
    :return: 旋转后的 [N, 7] 形式的边界框
    """
    rotated_bboxes = np.zeros_like(bboxes_3d)
    for i, bbox in enumerate(bboxes_3d):
        # 解包边界框的属性
        x, y, z, l, w, h, yaw = bbox
        
        # 将中心点位置转换为齐次坐标
        center_point = np.array([x, y, z, 1])
        
        # 仅应用旋转到中心点位置（不考虑z轴）
        center_point_rotated = np.dot(rotation_matrix, center_point[:3])
        
        # 计算新的yaw朝向
        # 假设旋转矩阵的二维旋转部分对应于yaw旋转
        yaw_rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        yaw_rotated = yaw - yaw_rotation
        
        # 更新旋转后的边界框属性
        rotated_bboxes[i] = [center_point_rotated[0], center_point_rotated[1], z, l, w, h, yaw_rotated]
        
    return rotated_bboxes


def transform_points(point_clouds, lidar2cam_r, lidar2cam_t):
    """
    Transform point clouds from lidar coordinate system to camera coordinate system.
    
    Parameters:
    - point_clouds: An (N, 3) numpy array of points in lidar coordinate system.
    - lidar2cam_t: Translation vector (3,) from lidar to camera coordinate system.
    - lidar2cam_r: Rotation matrix (3, 3) from lidar to camera coordinate system.
    
    Returns:
    - transformed_points: An (N, 3) numpy array of points in camera coordinate system.
    """
    # Convert the rotation matrix and translation vector into a full (4, 4) transformation matrix
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[:3, :3] = lidar2cam_r
    transformation_matrix[:3, 3] = lidar2cam_t
    transformation_matrix[3, 3] = 1
    
    # Ensure point clouds are in homogeneous coordinates for the transformation
    N = point_clouds.shape[0]
    ones = np.ones((N, 1))
    points_homogeneous = np.hstack([point_clouds, ones])
    
    # Apply the transformation matrix to the point clouds
    transformed_points_homogeneous = points_homogeneous @ transformation_matrix.T  # Matrix multiplication
    
    # Convert back from homogeneous coordinates to standard coordinates
    transformed_points = transformed_points_homogeneous[:, :3]  # Discard the homogeneous coordinate
    
    return transformed_points

def bbox3d_trans(bboxes_3d):
    rots, rotc = bboxes_3d[:, -2], bboxes_3d[:, -1]
    # rot = torch.atan2(rots, rotc)
    rot = np.arctan2(rots, rotc)[:, np.newaxis]
    return np.concatenate([bboxes_3d[:, :6], rot], axis=1)

def project_boxes_to_image(lidar2camera, camera_intrinsics, gt_bboxes_3d, ax, img_width, img_height, img_aug_matrix=None, color='r'):
    """
    将3D框投影到图像上，并以立方体形式画出
    :param lidar2camera: 4x4的变换矩阵，将LiDAR坐标系转换到相机坐标系
    :param camera_intrinsics: 3x3的相机内参矩阵
    :param gt_bboxes_3d: N x 7的3D框，[x, y, z, l, w, h, yaw]
    :param ax: matplotlib的轴对象，用于绘制
    :param img_width: 图像宽度
    :param img_height: 图像高度
    """
    import numpy as np

    # 遍历每个3D框
    for bbox in gt_bboxes_3d:
        x, y, z, l, w, h, yaw = bbox

        # 构建3D框的8个角点（以车辆坐标系为例，注意坐标轴方向）
        x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])

        # 旋转矩阵（绕z轴旋转yaw角度）
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        # 将角点旋转并平移到LiDAR坐标系
        corners = np.vstack((x_corners, y_corners, z_corners))  # (3,8)
        corners_lidar = R @ corners
        corners_lidar[0, :] += x
        corners_lidar[1, :] += y
        corners_lidar[2, :] += z

        # 转换为齐次坐标
        ones = np.ones((1, corners_lidar.shape[1]))
        corners_lidar_hom = np.vstack((corners_lidar, ones))  # (4,8)

        # 使用lidar2camera矩阵将角点转换到相机坐标系
        corners_camera_hom = lidar2camera @ corners_lidar_hom  # (4,8)

        # 转换为非齐次坐标
        corners_camera = corners_camera_hom[:3, :] / corners_camera_hom[3, :]

        # 过滤位于相机后方的框（所有角点的z坐标都小于等于0）
        if np.all(corners_camera[2, :] <= 0):
            continue

        # 使用相机内参矩阵将3D点投影到2D图像平面
        img_pts_hom = camera_intrinsics @ corners_camera  # (3,8)


        # 转换为像素坐标
        img_pts = img_pts_hom[:3, :] / img_pts_hom[2, :]  # (2,8)
        if img_aug_matrix is not None:
            # img_pts_hom = img_aug_matrix[:3, :3] @ img_pts_hom + img_aug_matrix[:3, 3].reshape(3, 1)
            img_pts = img_aug_matrix[:3, :3] @ img_pts + img_aug_matrix[:3, 3].reshape(3, 1)
        img_pts = img_pts[:2, :]
        # 检查是否至少有一个投影点在图像范围内且深度为正
        valid_mask = (img_pts[0, :] >= 0) & (img_pts[0, :] < img_width) & \
                     (img_pts[1, :] >= 0) & (img_pts[1, :] < img_height) & \
                     (corners_camera[2, :] > 0)
        # if not np.any(valid_mask):
        #     continue

        # 定义立方体的12条边（由顶点索引组成）
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ]

        # 绘制每条边
        for edge in edges:
            idx1, idx2 = edge
            # 如果任何一个端点在相机后方（z <= 0），则不绘制这条边
            if corners_camera[2, idx1] <= 0 or corners_camera[2, idx2] <= 0:
                continue
            x1, y1 = img_pts[:, idx1]
            x2, y2 = img_pts[:, idx2]
            # 绘制线段
            # print(x1, y1, x2, y2)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)
            
# def draw_bev_box(ax, bboxes_3d, color='r', name_list=None, prefix_num=3):
#     """
#     绘制鸟瞰图视角的2D框。
#     :param ax: Matplotlib 的 axes 对象。
#     :param bboxes_3d: 形状为 N*7 的数组，表示 N 个 3D 边界框。
#     :param color: 绘制边界框的颜色。
#     """


#     for i, bbox in enumerate(bboxes_3d):
#         # 清除空框
#         if sum(bbox[:6]) == 0:
#             continue
#         if bbox.shape[0] == 7:
#             x, y, _, l, w, h, yaw = bbox
#         elif bbox.shape[0] == 8:
#             x, y, _, l, w, h, yaw, label = bbox
#         else:
#             raise ValueError('每个检测框的格式必须是 (x, y, z, l, w, h, yaw) 或 (x, y, z, l, w, h, yaw, label)')
        
#         # yaw = -yaw
#         # 计算边界框的四个角点
#         corner_1 = [x - l / 2, y - w / 2]
#         corner_2 = [x + l / 2, y - w / 2]
#         corner_3 = [x + l / 2, y + w / 2]
#         corner_4 = [x - l / 2, y + w / 2]

#         corners = np.array([corner_1, corner_2, corner_3, corner_4])
#         # 旋转角点
#         c, s = np.cos(yaw), np.sin(yaw)
#         R = np.array([[c, -s], [s, c]])
#         rotated_corners = np.dot(corners - np.array([x, y]), R) + np.array([x, y])

#         # 绘制边界框 以及对应label
#         poly = plt.Polygon(rotated_corners, edgecolor=color, fill=False)
#         if bbox.shape[0] == 8:
#             ax.text(x, y, str(int(label)), color=color, fontsize=12, verticalalignment='top', horizontalalignment='left')
#         if name_list is not None:
#             ax.text(x, y, name_list[i][:prefix_num], color=color, fontsize=12, verticalalignment='top', horizontalalignment='left')
#         ax.add_patch(poly)

import matplotlib.pyplot as plt
import numpy as np
def draw_bev_box(ax, bboxes_3d, name_list=None, prefix_num=3, show_id=False, color_fix=None,
#                  category_colors= {
#     'car': (0.8392, 0.1529, 0.1569),        # 红色
#     'truck': (1.0, 0.4980, 0.0549),             # 橙色
#     'bus': (0.1725, 0.6275, 0.1725),            # 绿色
#     'trailer': (0.1216, 0.4667, 0.7059),            # 蓝色
#     'construction_vehicle': (0.5804, 0.4039, 0.7412),  # 紫色
#     'pedestrian': (0.5490, 0.3373, 0.2941),     # 棕色
#     'bicycle': (0.8902, 0.4667, 0.7608),        # 粉色
#     'motorcycle': (0.4980, 0.4980, 0.4980),     # 灰色
#     'traffic_cone': (0.7373, 0.7412, 0.1333),   # 橄榄色
#     'barrier': (0.0902, 0.7451, 0.8118)         # 青色
# }
category_colors = {
    'car': (0.9, 0.2, 0.2),                  # 更亮的红色
    'truck': (1.0, 0.6, 0.2),                # 更亮的橙色
    'bus': (0.2, 0.8, 0.2),                  # 更亮的绿色
    'trailer': (0.2, 0.5, 0.9),              # 更亮的蓝色
    'construction_vehicle': (0.7, 0.5, 0.9), # 更亮的紫色
    'pedestrian': (0.6, 0.4, 0.3),           # 更亮的棕色
    'bicycle': (0.95, 0.6, 0.8),             # 更亮的粉色
    'motorcycle': (0.6, 0.6, 0.6),           # 更亮的灰色
    'traffic_cone': (0.8, 0.8, 0.2),         # 更亮的橄榄色
    'barrier': (0.2, 0.8, 0.9)               # 更亮的青色
}
):
    """
    绘制鸟瞰图视角的2D框，并显示每个检测框的朝向。
    :param ax: Matplotlib 的 axes 对象。
    :param bboxes_3d: 形状为 N*7 或 N*8 的数组，表示 N 个 3D 边界框。
    :param name_list: 名称列表，用于显示在框上。
    :param prefix_num: 显示名称的前缀字符数。
    :param show_id: 布尔值，如果为 True，则显示框的 id。
    :param category_colors: 字典，键为类别名，值为 RGB 颜色值（范围在 0-1 的浮点数元组）。
    """
    # 定义固定的类别列表和默认颜色映射
    predefined_categories = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                             'pedestrian', 'bicycle', 'motorcycle', 'traffic_cone', 'barrier']
    cmap = plt.get_cmap('tab10')  # 使用 Matplotlib 的 'tab10' 调色板
    # 如果未提供类别颜色，则使用默认颜色
    if category_colors is None:
        colors = {category: cmap(i % 10) for i, category in enumerate(predefined_categories)}
    else:
        colors = category_colors
        
    default_color = 'black'  # 如果类别不在预定义列表中，使用默认颜色

    for i, bbox in enumerate(bboxes_3d):
        # 跳过空框
        if np.sum(bbox[:6]) == 0:
            continue
        if bbox.shape[0] == 7:
            x, y, _, l, w, h, yaw = bbox
            label = None
        elif bbox.shape[0] == 8:
            x, y, _, l, w, h, yaw, label = bbox
        else:
            raise ValueError('每个检测框的格式必须是 (x, y, z, l, w, h, yaw) 或 (x, y, z, l, w, h, yaw, label)')
        
        # 计算边界框的四个角点
        corner_1 = [x - l / 2, y - w / 2]
        corner_2 = [x + l / 2, y - w / 2]
        corner_3 = [x + l / 2, y + w / 2]
        corner_4 = [x - l / 2, y + w / 2]
        corners = np.array([corner_1, corner_2, corner_3, corner_4])
        
        # 旋转角点
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        rotated_corners = np.dot((corners - np.array([x, y])), R.T) + np.array([x, y])

        # 获取当前类别的颜色
        if name_list is not None:
            category = name_list[i].strip().lower()  # 规范化类别名称
            color = colors.get(category, default_color)
        else:
            color = default_color
        if color_fix is not None:
            color = color_fix

        # 绘制边界框
        poly = plt.Polygon(rotated_corners, edgecolor=color, fill=False, linewidth=2.5)
        ax.add_patch(poly)

        # 绘制朝向
        # 计算朝向的终点坐标
        orientation_length = max(l, w) * 0.5  # 朝向线的长度，可根据需要调整
        end_x = x + orientation_length * np.cos(yaw)
        end_y = y + orientation_length * np.sin(yaw)
        # 绘制朝向线
        # ax.arrow(x, y, end_x - x, end_y - y, color=color, width=0.1, head_width=0.3, length_includes_head=True)
        ax.plot([x, end_x], [y, end_y], color=color, linewidth=0.1)

        # 绘制标签
        label_text = ""
        if label is not None:
            label_text += str(int(label)) + " "
        if name_list is not None:
            label_text += category[:prefix_num] + " "
        if show_id:
            label_text += f"ID: {i}"
      
def draw_bboxes(bbox_data, ax, color='r', label_pos='top-left', name_list=None, prefix_num=3):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # 遍历每个检测框并绘制
    for i, bbox in enumerate(bbox_data):
        if sum(bbox[:4]) == 0:
            continue
        label = None  # 初始化label为None，以便后面检查是否存在
        if bbox.shape[0] == 5:
            x_min, y_min, width, height, label = bbox
        elif bbox.shape[0] == 4:
            x_min, y_min, width, height = bbox
        else:
            raise ValueError('每个检测框的格式必须是 (x_min, y_min, width, height) 或 (x_min, y_min, width, height, label)')

        # 创建一个矩形框
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color, facecolor='none')
        # 将矩形框添加到图像中
        ax.add_patch(rect)

        # 如果存在label，根据label_pos参数绘制label
        if label is not None:
            if label_pos == 'top-left':
                label_x, label_y = x_min, y_min
            elif label_pos == 'bottom-right':
                label_x, label_y = x_min + width, y_min + height
            else:
                raise ValueError("label_pos参数必须是 'top-left' 或 'bottom-right'")
            
            # 绘制标签
            ax.text(label_x, label_y, str(int(label)), color=color, fontsize=12, verticalalignment='top' if label_pos == 'top-left' else 'bottom', horizontalalignment='left' if label_pos == 'top-left' else 'right')
        if name_list is not None:
            ax.text(x_min, y_min, name_list[i][:prefix_num], color=color, fontsize=2, verticalalignment='top', horizontalalignment='left')


    
def project_3d_to_2d_numpy(bboxes_3d, lidar2cam_t, lidar2cam_r, cam_intrinsic, img_shape, mode='center', diff=5, img_aug_matrix=None):
    assert mode in ['center', 'corners']
    
    def rotation_matrix(yaw):
        """Create a rotation matrix from yaw angle using NumPy."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array([[cos_yaw, sin_yaw, 0], 
                         [-sin_yaw, cos_yaw, 0], 
                         [0, 0, 1]])

    projected_2d_boxes = []
    valid_index = []
    img_height, img_width = img_shape[:2]
    for i, bbox in enumerate(bboxes_3d):
        if sum(bbox[:6]) == 0:
            continue
        if bbox.shape[0] == 8:
            x, y, z, l, w, h, yaw, label = bbox
        elif bbox.shape[0] == 7:
            x, y, z, l, w, h, yaw = bbox
        else:
            raise ValueError('每个检测框的格式必须是 (x, y, z, l, w, h, yaw) 或 (x, y, z, l, w, h, yaw, label)')
        R_yaw = rotation_matrix(yaw)
        
        corners = np.array([
            [l/2, w/2, h/2, 1], [-l/2, w/2, h/2, 1],
            [-l/2, -w/2, h/2, 1], [l/2, -w/2, h/2, 1],
            [l/2, w/2, -h/2, 1], [-l/2, w/2, -h/2, 1],
            [-l/2, -w/2, -h/2, 1], [l/2, -w/2, -h/2, 1]
        ]).T
        
        corners_rotated = R_yaw @ corners[:3, :]
        corners_lidar = corners_rotated + np.array([x, y, z]).reshape(3, 1)
        corners_cam = lidar2cam_r @ corners_lidar + lidar2cam_t.reshape(3, 1)
        
        if np.all(corners_cam[2, :] < 0):
            continue
        

        corners_image = cam_intrinsic @ corners_cam[:3, :]
        positive_index = corners_image[2, :] > 0
        corners_image /= corners_image[2, :]  # 归一化
        if img_aug_matrix is not None:
            corners_image = img_aug_matrix[:3, :3] @ corners_image + img_aug_matrix[:3, 3].reshape(3, 1)
        
        x_min, y_min = np.min(corners_image[:2, positive_index], axis=1)
        x_max, y_max = np.max(corners_image[:2, positive_index], axis=1)

        if x_min > img_width or y_min > img_height or x_max < 0 or y_max < 0:
                continue
        if x_min > img_width - diff or y_min > img_height - diff or x_max < 0 + diff or y_max < 0 + diff:
            continue
        if mode == 'center': 
            # 计算中心点并检查是否在图像内
            center_image = np.mean(corners_image[:2, :], axis=1)
            if not (0 <= center_image[0] <= img_width and 0 <= center_image[1] <= img_height):
                continue
        
        elif mode == 'corners':
            # 过滤掉完全位于图像外部的检测框
            x_min, y_min = np.min(corners_image[:2, :], axis=1)
            x_max, y_max = np.max(corners_image[:2, :], axis=1)
            if x_min > img_width - diff or y_min > img_height - diff or x_max < 0 + diff or y_max < 0 + diff:
                continue
        x_min, y_min = np.min(corners_image[:2, :], axis=1)
        x_max, y_max = np.max(corners_image[:2, :], axis=1)
        # 调整检测框坐标，确保它们位于图像内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)
        valid_index.append(i)
        if bboxes_3d.shape[1] == 8:
            projected_2d_boxes.append(np.array([x_min, y_min, x_max - x_min, y_max - y_min, label]))
        else:
            projected_2d_boxes.append(np.array([x_min, y_min, x_max - x_min, y_max - y_min]))
    
    if projected_2d_boxes != []:
        return np.stack(projected_2d_boxes), valid_index
    else:
        if bboxes_3d.shape[1] == 8:
            return np.empty((0, 5)), np.empty((0, 1))
        else:
            return np.empty((0, 4)), np.empty((0, 1))
def project_3d_to_2d_torch(bboxes_3d, lidar2cam_t, lidar2cam_r, cam_intrinsic, img_shape, mode='center', diff=5):
    assert mode in ['center', 'corners']
    
    def rotation_matrix(yaw):
        """Create a rotation matrix from yaw angle using torch."""
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        return torch.tensor([[cos_yaw, -sin_yaw, 0], 
                             [sin_yaw, cos_yaw, 0], 
                             [0, 0, 1]], device=bboxes_3d.device)

    # Convert numpy arrays to torch tensors
    lidar2cam_t = torch.tensor(lidar2cam_t, dtype=torch.float32, device=bboxes_3d.device)
    lidar2cam_r = torch.tensor(lidar2cam_r, dtype=torch.float32, device=bboxes_3d.device)
    cam_intrinsic = torch.tensor(cam_intrinsic, dtype=torch.float32, device=bboxes_3d.device)
    valid_mask = torch.zeros_like(bboxes_3d[:, 0], dtype=torch.bool, device=bboxes_3d.device)
    
    projected_2d_boxes = []
    img_height, img_width = img_shape[:2]
    for i, bbox in enumerate(bboxes_3d):
        if torch.sum(bbox[:6]) == 0:
            continue
        if bbox.shape[0] == 8:
            x, y, z, l, w, h, yaw, label = bbox
        elif bbox.shape[0] == 7:
            x, y, z, l, w, h, yaw = bbox
        else:
            raise ValueError('Each bounding box must be formatted as (x, y, z, l, w, h, yaw) or (x, y, z, l, w, h, yaw, label)')
        
        R_yaw = rotation_matrix(-yaw) # [3, 3]
        
        corners = torch.tensor([
            [l/2, w/2, h/2, 1], [-l/2, w/2, h/2, 1],
            [-l/2, -w/2, h/2, 1], [l/2, -w/2, h/2, 1],
            [l/2, w/2, -h/2, 1], [-l/2, w/2, -h/2, 1],
            [-l/2, -w/2, -h/2, 1], [l/2, -w/2, -h/2, 1]
        ], dtype=torch.float32, device=bboxes_3d.device).T # [4, 8]
        
        corners_rotated = R_yaw @ corners[:3, :] # [3, 3] * [3, 8] = [3, 8]
        corners_lidar = corners_rotated + torch.tensor([x, y, z], device=bboxes_3d.device).reshape(3, 1) # [3, 8] + [3, 1] = [3, 8]
        corners_cam = lidar2cam_r @ corners_lidar + lidar2cam_t.reshape(3, 1) # [3, 3] * [3, 8] + [3, 1] = [3, 8]
        
        if torch.all(corners_cam[2, :] < 0):
            continue
        
        corners_image = cam_intrinsic @ corners_cam[:3, :] # [3, 3] * [3, 8] = [3, 8]
        positive_index = corners_image[2, :] > 0
        # corners_image /= corners_image[2, :].unsqueeze(0)  # Normalization
        corners_image = corners_image / corners_image[2, :].unsqueeze(0) # [3, 8] / [1, 8] = [3, 8]
        
        x_min, y_min = torch.min(corners_image[:2, positive_index], dim=1).values
        x_max, y_max = torch.max(corners_image[:2, positive_index], dim=1).values

        if x_min > img_width or y_min > img_height or x_max < 0 or y_max < 0:
                continue
        if x_min > img_width - diff or y_min > img_height - diff or x_max < diff or y_max < diff:
            continue
        
        x_min = max(0, x_min.item())
        y_min = max(0, y_min.item())
        x_max = min(img_width, x_max.item())
        y_max = min(img_height, y_max.item())
        
        valid_mask[i] = True
        if bbox.shape[0] == 8:
            projected_2d_boxes.append(torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min, label], device=bboxes_3d.device))
        else:
            projected_2d_boxes.append(torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], device=bboxes_3d.device))
    
    if projected_2d_boxes:
        return torch.stack(projected_2d_boxes), valid_mask
    else:
        if bboxes_3d.shape[1] == 8:
            return torch.empty((0, 5), device=bboxes_3d.device), valid_mask
        else:
            return torch.empty((0, 4), device=bboxes_3d.device), valid_mask

def get_conners(bboxes_3d):
    bboxes_3d = bboxes_3d.float()
    cos_yaw = torch.cos(-bboxes_3d[:, 6])
    sin_yaw = torch.sin(-bboxes_3d[:, 6])
    zero = torch.zeros_like(cos_yaw)
    one = torch.ones_like(cos_yaw)
    R_yaw = torch.stack([cos_yaw, -sin_yaw, zero, sin_yaw, cos_yaw, zero, zero, zero, one], dim=-1).view(-1, 3, 3)

    l, w, h = bboxes_3d[:, 3], bboxes_3d[:, 4], bboxes_3d[:, 5]
    corners_base = torch.tensor([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]], device=bboxes_3d.device)
    corners = corners_base[None, :, :] * torch.stack([l, w, h], dim=-1)[:, None, :]
    corners = corners.permute(0, 2, 1) # [200, 3, 8]

    x, y, z = bboxes_3d[:, 0], bboxes_3d[:, 1], bboxes_3d[:, 2] # [200] [200] [200]
    corners_rotated = torch.bmm(R_yaw, corners) # [200, 3, 8]
    corners_lidar = corners_rotated + torch.stack([x, y, z], dim=-1).unsqueeze(-1) # [200, 3, 8] + [200, 3, 1] = [200, 3, 8] # 得到lidar坐标系下的8个角点
    return corners_lidar
def project_3d_to_2d_torch_parallel_opt(corners_lidar, lidar2pix, img_shape, diff=5):
    img_height, img_width = img_shape[:2]

    corners_image = torch.bmm(lidar2pix, corners_lidar) # [1200, 4, 4] * [1200, 4, 8] = [1200, 4, 8]
    corners_image = corners_image[:, :3, :] # [1200, 3, 8]
    valid_points_mask = corners_image[:, 2, :] > 0 # [1200, 8]
    valid_boxes_mask = valid_points_mask.any(dim=1)
    valid_points_mask_expanded = valid_points_mask.unsqueeze(1)
    # true_indices_per_row = []
    # for i in range(valid_points_mask.shape[0]):
    #     true_indices_per_row.append(torch.where(valid_points_mask[i])[0])

    corners_image = corners_image / (corners_image[:, 2, :].unsqueeze(1) + 1e-4)  # [1200, 3, 8] / [1200, 1, 8] = [1200, 3, 8]
    corners_image_masked_max = torch.where(valid_points_mask_expanded, corners_image, torch.tensor(float('-inf')).to(corners_image.device))
    corners_image_masked_min = torch.where(valid_points_mask_expanded, corners_image, torch.tensor(float('inf')).to(corners_image.device))
    
    x_min, _ = corners_image_masked_min[:, 0, :].min(dim=1)
    y_min, _ = corners_image_masked_min[:, 1, :].min(dim=1)
    x_max, _ = corners_image_masked_max[:, 0, :].max(dim=1)
    y_max, _ = corners_image_masked_max[:, 1, :].max(dim=1)

    valid_boxes_mask2 = (x_min <= img_width - diff) & (y_min <= img_height - diff) & \
        (x_max >= diff) & (y_max >= diff) & valid_boxes_mask
    
    # 裁剪坐标确保它们在图像内
    x_min = torch.clamp(x_min, min=0)
    y_min = torch.clamp(y_min, min=0)
    x_max = torch.clamp(x_max, max=img_width)
    y_max = torch.clamp(y_max, max=img_height)

    projected_2d_boxes = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=1)
    
    projected_2d_boxes_list = []
    valid_boxes_mask_list = []
    for i in range(6):
        temp_valid_boxes_mask = valid_boxes_mask2[i*200:(i+1)*200]
        projected_2d_boxes_list.append(projected_2d_boxes[i*200:(i+1)*200][temp_valid_boxes_mask])
        valid_boxes_mask_list.append(temp_valid_boxes_mask)
    
    return projected_2d_boxes_list, valid_boxes_mask_list

def project_3d_to_2d_torch_parallel(bboxes_3d, lidar2cam_t, lidar2cam_r, cam_intrinsic, img_shape, diff=5):
    bboxes_3d = bboxes_3d.float()
    lidar2cam_t = torch.tensor(lidar2cam_t, dtype=torch.float32, device=bboxes_3d.device).unsqueeze(0)
    lidar2cam_r = torch.tensor(lidar2cam_r, dtype=torch.float32, device=bboxes_3d.device)
    cam_intrinsic = torch.tensor(cam_intrinsic, dtype=torch.float32, device=bboxes_3d.device)

    img_height, img_width = img_shape[:2]

    cos_yaw = torch.cos(-bboxes_3d[:, 6])
    sin_yaw = torch.sin(-bboxes_3d[:, 6])
    zero = torch.zeros_like(cos_yaw)
    one = torch.ones_like(cos_yaw)
    R_yaw = torch.stack([cos_yaw, -sin_yaw, zero, sin_yaw, cos_yaw, zero, zero, zero, one], dim=-1).view(-1, 3, 3)

    l, w, h = bboxes_3d[:, 3], bboxes_3d[:, 4], bboxes_3d[:, 5]
    corners_base = torch.tensor([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]], device=bboxes_3d.device)
    corners = corners_base[None, :, :] * torch.stack([l, w, h], dim=-1)[:, None, :]
    corners = corners.permute(0, 2, 1) # [200, 3, 8]

    x, y, z = bboxes_3d[:, 0], bboxes_3d[:, 1], bboxes_3d[:, 2] # [200] [200] [200]
    corners_rotated = torch.bmm(R_yaw, corners) # [200, 3, 8]
    corners_lidar = corners_rotated + torch.stack([x, y, z], dim=-1).unsqueeze(-1) # [200, 3, 8] + [200, 3, 1] = [200, 3, 8]
    
    # lidar2cam_r 复制corners_lidar.shape[0]个
    lidar2cam_r_repeat = lidar2cam_r.repeat(bboxes_3d.shape[0], 1, 1) # [200, 3, 3]
    lidar2cam_t_repeat = lidar2cam_t.permute(1, 0).repeat(bboxes_3d.shape[0], 1, 1) # [200, 3, 1]
    cam_intrinsic_repeat = cam_intrinsic.repeat(bboxes_3d.shape[0], 1, 1) # [200, 3, 3]
    corners_cam = torch.bmm(lidar2cam_r_repeat, corners_lidar) + lidar2cam_t_repeat # [200, 3, 3] * [200, 3, 8] + [200, 3, 1] = [200, 3, 8]

    corners_image = torch.bmm(cam_intrinsic_repeat, corners_cam) # [200, 3, 3] * [200, 3, 8] = [200, 3, 8]
    valid_points_mask = corners_image[:, 2, :] > 0 # [200, 8]
    valid_boxes_mask = valid_points_mask.any(dim=1)
    true_indices_per_row = []
    for i in range(valid_points_mask.shape[0]):
        true_indices_per_row.append(torch.where(valid_points_mask[i])[0])

    corners_image = corners_image / corners_image[:, 2, :].unsqueeze(1)  # [200, 3, 8] / [200, 1, 8] = [200, 3, 8]

    projected_2d_boxes = torch.full((bboxes_3d.size(0), 4), float('inf'), device=bboxes_3d.device) # [200, 4]
    
    for i in range(bboxes_3d.size(0)):
        if valid_boxes_mask[i]:
            valid_points = corners_image[i][:2, true_indices_per_row[i]]
            x_min, y_min = valid_points.min(dim=1).values
            x_max, y_max = valid_points.max(dim=1).values

            # if x_min > img_width or y_min > img_height or x_max < 0 or y_max < 0:
            #     continue
            if x_min > img_width - diff or y_min > img_height - diff or x_max < diff or y_max < diff:
                valid_boxes_mask[i] = False
                continue
        
            x_min = max(0, x_min.item())
            y_min = max(0, y_min.item())
            x_max = min(img_width, x_max.item())
            y_max = min(img_height, y_max.item())

            projected_2d_boxes[i] = torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], device=bboxes_3d.device)

    projected_2d_boxes = projected_2d_boxes[valid_boxes_mask]

    return projected_2d_boxes, valid_boxes_mask
def project_3d_to_2d_torch_parallel_float16(bboxes_3d, lidar2cam_t, lidar2cam_r, cam_intrinsic, img_shape, diff=5):
    # 转换为float16进行处理
    bboxes_3d = bboxes_3d.float().to(dtype=torch.float16)
    lidar2cam_t = torch.tensor(lidar2cam_t, dtype=torch.float16, device=bboxes_3d.device).unsqueeze(0)
    lidar2cam_r = torch.tensor(lidar2cam_r, dtype=torch.float16, device=bboxes_3d.device)
    cam_intrinsic = torch.tensor(cam_intrinsic, dtype=torch.float16, device=bboxes_3d.device)

    img_height, img_width = img_shape[:2]

    cos_yaw = torch.cos(-bboxes_3d[:, 6])
    sin_yaw = torch.sin(-bboxes_3d[:, 6])
    zero = torch.zeros_like(cos_yaw)
    one = torch.ones_like(cos_yaw)
    R_yaw = torch.stack([cos_yaw, -sin_yaw, zero, sin_yaw, cos_yaw, zero, zero, zero, one], dim=-1).view(-1, 3, 3)

    l, w, h = bboxes_3d[:, 3], bboxes_3d[:, 4], bboxes_3d[:, 5]
    corners_base = torch.tensor([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], 
                                 [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]], 
                                dtype=torch.float16, device=bboxes_3d.device)
    corners = corners_base[None, :, :] * torch.stack([l, w, h], dim=-1)[:, None, :]
    corners = corners.permute(0, 2, 1)

    x, y, z = bboxes_3d[:, 0], bboxes_3d[:, 1], bboxes_3d[:, 2]
    corners_rotated = torch.bmm(R_yaw, corners)
    corners_lidar = corners_rotated + torch.stack([x, y, z], dim=-1).unsqueeze(-1)
    
    lidar2cam_r_repeat = lidar2cam_r.repeat(bboxes_3d.shape[0], 1, 1)
    lidar2cam_t_repeat = lidar2cam_t.permute(1, 0).repeat(bboxes_3d.shape[0], 1, 1)
    cam_intrinsic_repeat = cam_intrinsic.repeat(bboxes_3d.shape[0], 1, 1)
    corners_cam = torch.bmm(lidar2cam_r_repeat, corners_lidar) + lidar2cam_t_repeat

    corners_image = torch.bmm(cam_intrinsic_repeat, corners_cam)
    valid_points_mask = corners_image[:, 2, :] > 0
    valid_boxes_mask = valid_points_mask.any(dim=1)

    corners_image = corners_image / corners_image[:, 2, :].unsqueeze(1)

    projected_2d_boxes = torch.full((bboxes_3d.size(0), 4), float('inf'), dtype=torch.float16, device=bboxes_3d.device)
    
    for i in range(bboxes_3d.size(0)):
        if valid_boxes_mask[i]:
            valid_points = corners_image[i][:2, valid_points_mask[i]]
            x_min, y_min = valid_points.min(dim=1).values
            x_max, y_max = valid_points.max(dim=1).values

            if x_min > img_width - diff or y_min > img_height - diff or x_max < diff or y_max < diff:
                valid_boxes_mask[i] = False
                continue
        
            x_min = torch.max(torch.tensor(0.0, dtype=torch.float16, device=bboxes_3d.device), x_min)
            y_min = torch.max(torch.tensor(0.0, dtype=torch.float16, device=bboxes_3d.device), y_min)
            x_max = torch.min(torch.tensor(img_width, dtype=torch.float16, device=bboxes_3d.device), x_max)
            y_max = torch.min(torch.tensor(img_height, dtype=torch.float16, device=bboxes_3d.device), y_max)

            projected_2d_boxes[i] = torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], dtype=torch.float16, device=bboxes_3d.device)

    projected_2d_boxes = projected_2d_boxes[valid_boxes_mask]

    return projected_2d_boxes, valid_boxes_mask
def bboxes3d_decoder_torch_parallel(bboxes, out_size_factor=8, voxel_size=[0.075, 0.075], pc_range=[-54.0, -54.0]):
    # 确保输入为torch张量
    bboxes = bboxes.float()
    
    # voxel_size 和 pc_range 转为torch张量
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=bboxes.device)
    pc_range = torch.tensor(pc_range, dtype=torch.float32, device=bboxes.device)
    
    # 分别处理每个部分
    # 将x, y转换为实际世界的度量
    x = bboxes[:, 0] * out_size_factor * voxel_size[0] + pc_range[0]
    y = bboxes[:, 1] * out_size_factor * voxel_size[1] + pc_range[1]
    z = bboxes[:, 2]
    
    # 将l, w, h从对数空间转换回来
    l = torch.exp(bboxes[:, 3])
    w = torch.exp(bboxes[:, 4])
    h = torch.exp(bboxes[:, 5])
    
    # 计算旋转
    rot = torch.atan2(bboxes[:, 6], bboxes[:, 7])

    # 准备解码后的bbox
    decoded_bboxes = torch.stack([x, y, z, l, w, h, rot], dim=1)
    
    # 如果提供了速度信息，则包含在解码后的输出中
    if bboxes.shape[1] > 8:
        decoded_bboxes = torch.cat((decoded_bboxes, bboxes[:, 8:]), dim=1)
    
    return decoded_bboxes
def calculate_iou_numpy(bboxes_2d, bboxes_projected, iou_type='iou'):
    """
    Calculate the IoU between 2D bboxes and their projected 2D bboxes.
    Args:
        bboxes_2d: 2D bboxes in format N * (x_min, y_min, x_max, y_max, label)
        bboxes_projected: Projected 2D bboxes in format M * (x_min, y_min, x_max, y_max, label)
        iou_type: The type of IoU to calculate. Can be 'iou' or 'giou'.
    Returns:
        iou: IoU between bboxes_2d and bboxes_projected, in format N * M
    """
    assert iou_type in ['iou', 'giou'], "iou_type must be 'iou' or 'giou'"
    
    iou_matrix = np.zeros((bboxes_2d.shape[0], bboxes_projected.shape[0]))
    
    for i, box1 in enumerate(bboxes_2d):
        for j, box2 in enumerate(bboxes_projected):
            # 检查标签是否相同
            if box1[-1] != box2[-1]:
                continue  # 如果标签不同，跳过当前的计算
                
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            unionArea = box1Area + box2Area - interArea
            
            iou = interArea / unionArea
            
            if iou_type == 'iou':
                iou_matrix[i, j] = iou
            elif iou_type == 'giou':
                xC = min(box1[0], box2[0])
                yC = min(box1[1], box2[1])
                xD = max(box1[2], box2[2])
                yD = max(box1[3], box2[3])
                closureArea = (xD - xC) * (yD - yC)
                giou = iou - (closureArea - unionArea) / closureArea
                iou_matrix[i, j] = giou
                
    return iou_matrix

def calculate_iou_torch(bboxes_2d, bboxes_projected, iou_type='iou', use_label=True):
    """
    Calculate the IoU between 2D bboxes and their projected 2D bboxes using PyTorch.
    Args:
        bboxes_2d: 2D bboxes in format N * (x_min, y_min, x_max, y_max, label), torch.Tensor
        bboxes_projected: Projected 2D bboxes in format M * (x_min, y_min, x_max, y_max, label), torch.Tensor
        iou_type: The type of IoU to calculate. Can be 'iou' or 'giou'.
    Returns:
        iou: IoU between bboxes_2d and bboxes_projected, in format N * M, torch.Tensor
    """
    assert iou_type in ['iou', 'giou'], "iou_type must be 'iou' or 'giou'"
    
    iou_matrix = torch.zeros((bboxes_2d.shape[0], bboxes_projected.shape[0]), device=bboxes_2d.device)
    
    for i, box1 in enumerate(bboxes_2d):
        for j, box2 in enumerate(bboxes_projected):
            # Check if labels match
            if use_label and box1[-1] != box2[-1]:
                continue  # Skip if labels do not match
            
            xA = torch.max(box1[0], box2[0])
            yA = torch.max(box1[1], box2[1])
            xB = torch.min(box1[2], box2[2])
            yB = torch.min(box1[3], box2[3])
            
            interArea = torch.max(torch.tensor(0., device=bboxes_2d.device), xB - xA) * torch.max(torch.tensor(0., device=bboxes_2d.device), yB - yA)
            box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            unionArea = box1Area + box2Area - interArea
            
            iou = interArea / unionArea
            
            if iou_type == 'iou':
                iou_matrix[i, j] = iou
            elif iou_type == 'giou':
                xC = torch.min(box1[0], box2[0])
                yC = torch.min(box1[1], box2[1])
                xD = torch.max(box1[2], box2[2])
                yD = torch.max(box1[3], box2[3])
                closureArea = (xD - xC) * (yD - yC)
                giou = iou - (closureArea - unionArea) / closureArea
                iou_matrix[i, j] = giou
                
    return iou_matrix

def denormalize(img, img_norm_cfg):
    mean = img_norm_cfg['mean']
    std = img_norm_cfg['std']
    to_rgb = img_norm_cfg['to_rgb']
    
    # 反归一化
    img = img * std[:, None, None] + mean[:, None, None]
    
    # 如果需要将颜色通道从BGR转换为RGB
    if to_rgb:
        img = img[::-1, :, :]
    
    # 确保图像数据类型是uint8，便于可视化
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def bboxes3d_decoder(bboxes, out_size_factor=8, voxel_size=[0.075, 0.075], pc_range=[-54.0, -54.0]):
    # 确认输入为numpy数组
    bboxes = np.asarray(bboxes, dtype=np.float32)
    
    # 预处理输出列表
    decoded_bboxes = []
    
    # 遍历并解码每个bbox
    for bbox in bboxes:
        # 基本参数
        x, y, z, l, w, h, rots, rotc = bbox[:8]
        
        # 将x, y转换为实际世界的度量
        x = x * out_size_factor * voxel_size[0] + pc_range[0]
        y = y * out_size_factor * voxel_size[1] + pc_range[1]
        
        # 将l, w, h从对数空间转换回来，如果适用
        l, w, h = np.exp(l), np.exp(w), np.exp(h)
        # z = z - h * 0.5  # 从中心到底部
        
        # 计算旋转
        rot = np.arctan2(rots, rotc)
        
        # 准备解码后的bbox
        decoded_bbox = [x, y, z, l, w, h, rot]
        
        # 如果提供了速度信息，则包含在解码后的输出中
        if len(bbox) > 8:
            vel = bbox[8:]
            decoded_bbox.extend(vel)
        
        decoded_bboxes.append(decoded_bbox)
    
    return np.array(decoded_bboxes)


def bboxes3d_decoder_torch(bboxes, out_size_factor=8, voxel_size=[0.075, 0.075], pc_range=[-54.0, -54.0]):
    # 确保输入为torch张量
    bboxes = bboxes.float()
    
    # voxel_size 和 pc_range 转为torch张量
    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=bboxes.device)
    pc_range = torch.tensor(pc_range, dtype=torch.float32, device=bboxes.device)
    
    # 预处理输出列表
    decoded_bboxes = []
    
    # 遍历并解码每个bbox
    for bbox in bboxes:
        # 基本参数
        x, y, z, l, w, h, rots, rotc = bbox[:8]
        
        # 将x, y转换为实际世界的度量
        x = x * out_size_factor * voxel_size[0] + pc_range[0]
        y = y * out_size_factor * voxel_size[1] + pc_range[1]
        
        # 将l, w, h从对数空间转换回来
        l, w, h = torch.exp(l), torch.exp(w), torch.exp(h)
        # z = z - h * 0.5  # 从中心到底部，如果需要调整z值
        
        # 计算旋转
        rot = torch.atan2(rots, rotc)
        
        # 准备解码后的bbox
        decoded_bbox = torch.tensor([x, y, z, l, w, h, rot], device=bboxes.device)
        
        # 如果提供了速度信息，则包含在解码后的输出中
        if len(bbox) > 8:
            vel = bbox[8:]
            decoded_bbox = torch.cat((decoded_bbox, vel))
        
        decoded_bboxes.append(decoded_bbox)
    
    # 将解码后的bboxes列表转换为一个torch张量
    return torch.stack(decoded_bboxes)

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou_with_eps(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-4) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter + eps)
    return iou


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        generalized_iou (Tensor[N, M]): the NxM matrix containing the pairwise generalized_IoU values
        for every element in boxes1 and boxes2
    """

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def adjust_yaw_parallel(yaws, lidar2cam_r):
    """
    Adjust an array of yaw angles based on the rotation matrix using NumPy.
    This is a simplified example.
    """
    def limit_period(val, offset=0.5, period=np.pi):
        """Limit the value into a period for periodic function using NumPy.

        Args:
            val (np.ndarray): The value to be converted.
            offset (float, optional): Offset to set the value range. Defaults to 0.5.
            period (float, optional): Period of the value. Defaults to np.pi.

        Returns:
            np.ndarray: Value in the range of [-offset * period, (1-offset) * period]
        """
        return val - np.floor(val / period + offset) * period
    
    # yaw_view = -yaws - np.pi / 2
    yaw_view = yaws
    rot_dir_view = np.concatenate([np.cos(yaw_view)[:, None], np.sin(yaw_view)[:, None], np.zeros_like(yaw_view)[:, None]], axis=1) # (N, 3)
    rot_dir_view = np.dot(rot_dir_view, lidar2cam_r.T) # (N, 3)
    rot_dir_view = rot_dir_view[:, [0, 2]] # (N, 2)
    yaw_view = -np.arctan2(rot_dir_view[:, 1], rot_dir_view[:, 0]) # (N,)
    # yaw_view = limit_period(yaw_view, period=2 * np.pi) # (N,)

    return yaw_view


def transform_velocity_parallel(vxs, vys, lidar2cam_r):
    """
    Transform arrays of velocity components from lidar to camera coordinate system. 
    This example considers only rotation.
    """
    velocities_lidar = np.vstack([vxs, vys, np.zeros_like(vxs)])
    velocities_camera = lidar2cam_r @ velocities_lidar
    return velocities_camera[0, :], velocities_camera[1, :]



def transform_lidar_to_camera_parallel(detection_boxes_lidar, lidar2cam_r, lidar2cam_t):
    """
    Transform 3D detection boxes from lidar coordinate system to camera coordinate system,
    including adjustments for yaw angle and velocity, in a parallel manner.
    """
    # Separate components for clarity
    box_centers = detection_boxes_lidar[:, :3]
    dimensions = detection_boxes_lidar[:, 3:6]  # l, w, h
    yaws = detection_boxes_lidar[:, 6]
    if detection_boxes_lidar.shape[1] > 8:
        velocities = detection_boxes_lidar[:, 7:9]  # vx, vy
    
    # Convert box centers to homogeneous coordinates
    ones = np.ones((box_centers.shape[0], 1))
    box_centers_homogeneous = np.hstack([box_centers, ones])
    lidar2cam_mat = np.hstack([lidar2cam_r, lidar2cam_t[:, np.newaxis]])
    lidar2cam_mat = np.vstack([lidar2cam_mat, [0, 0, 0, 1]])
    # Apply rotation and translation to box centers
    centers_camera_homogeneous = lidar2cam_mat @ box_centers_homogeneous.T
    centers_camera = centers_camera_homogeneous[:3, :].T
    
    # Adjust yaws and velocities
    yaws_adjusted = adjust_yaw_parallel(yaws, lidar2cam_r)
    
    # Concatenate results into the final array
    if detection_boxes_lidar.shape[1] > 8:
        vx_cam, vy_cam = transform_velocity_parallel(velocities[:, 0], velocities[:, 1], lidar2cam_r)
        detection_boxes_camera = np.hstack([centers_camera, dimensions, yaws_adjusted[:, np.newaxis], vx_cam[:, np.newaxis], vy_cam[:, np.newaxis]])
    else:
        detection_boxes_camera = np.hstack([centers_camera, dimensions, yaws_adjusted[:, np.newaxis]])
    
    return detection_boxes_camera


def project_lidar_to_image(points, lidar2camera, camera_intrinsics, image_aug_matrix, img_width, img_height):
    """
    将 LiDAR 点投影到图像平面，并应用图像增强矩阵，同时过滤掉不在图像范围内的点和深度为负数的点。
    :param points: LiDAR 坐标系下的点云，大小为 [N, 3]，每行是 (X, Y, Z) 坐标
    :param lidar2camera: LiDAR 到相机的外参矩阵，大小为 [4, 4]
    :param camera_intrinsics: 相机的内参矩阵，大小为 [4, 4]
    :param image_aug_matrix: 图像增强矩阵，大小为 [4, 4]
    :param img_width: 图像的宽度（像素）
    :param img_height: 图像的高度（像素）
    :return: 过滤后的图像平面上的点，大小为 [N, 2]，表示像素坐标
    """
    
    # 1. 将点云扩展为齐次坐标 (X, Y, Z, 1)，以便与 4x4 矩阵相乘
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # [N, 4]
    
    # 2. 使用 lidar2camera 矩阵将 LiDAR 坐标系下的点转换为相机坐标系下的点
    points_camera = np.dot(lidar2camera, points_homogeneous.T).T  # [N, 4]
    
    # 3. 从相机坐标系中提取 X, Y, Z 坐标（齐次坐标，第四列被丢弃）
    points_camera = points_camera[:, :3]  # [N, 3]

    # 4. 过滤掉深度 Z_c 小于等于 0 的点
    valid_depth_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_depth_mask]  # 只保留深度为正的点
    
    # 5. 将相机坐标系下的点归一化 (X_c / Z_c, Y_c / Z_c)，并添加一个 1，形成齐次坐标
    points_camera_normalized = np.hstack([points_camera[:, :2] / points_camera[:, 2:3], np.ones((points_camera.shape[0], 1))])  # [N, 3]
    
    # 6. 使用相机内参矩阵将归一化坐标转换为图像上的像素坐标
    points_image_homogeneous = np.dot(camera_intrinsics, np.hstack([points_camera_normalized, np.ones((points_camera_normalized.shape[0], 1))]).T).T  # [N, 4]

    # 7. 应用图像增强矩阵进行变换
    points_image_augmented = np.dot(image_aug_matrix, points_image_homogeneous.T).T  # [N, 4]
    
    # 8. 执行透视除法，将齐次坐标转换为 2D 像素坐标
    points_image = points_image_augmented[:, :2] / points_image_augmented[:, 3:4]  # [N, 2]
    
    # 9. 过滤掉不在图像范围内的点
    valid_x_mask = (points_image[:, 0] >= 0) & (points_image[:, 0] < img_width)  # 保留在 [0, img_width) 范围内的 u 坐标
    valid_y_mask = (points_image[:, 1] >= 0) & (points_image[:, 1] < img_height)  # 保留在 [0, img_height) 范围内的 v 坐标
    valid_mask = valid_x_mask & valid_y_mask  # 同时满足 u 和 v 坐标范围的点
    
    # 10. 应用过滤条件，得到在图像范围内的有效点
    points_image_filtered = points_image[valid_mask]
    points_depth_filtered = points_camera[valid_mask][:, 2]  # 保留对应的深度信息
    
    # 返回过滤后的图像上的 2D 像素坐标和深度信息
    return points_image_filtered, points_depth_filtered





def project_lidar_to_image_torch(points, lidar2camera, camera_intrinsics, image_aug_matrix, img_width, img_height, lidar_aug_matrix=None):
    """
    将 LiDAR 点投影到图像平面，并应用图像增强矩阵和可选的 LiDAR 数据增强矩阵，
    同时过滤掉不在图像范围内的点和深度为负数的点。
    
    :param points: LiDAR 坐标系下的点云，大小为 [N, 3]，每行是 (X, Y, Z) 坐标
    :param lidar2camera: LiDAR 到相机的外参矩阵，大小为 [4, 4]
    :param camera_intrinsics: 相机的内参矩阵，大小为 [4, 4]
    :param image_aug_matrix: 图像增强矩阵，大小为 [4, 4]
    :param img_width: 图像的宽度（像素）
    :param img_height: 图像的高度（像素）
    :param lidar_aug_matrix: LiDAR 点云增强矩阵 (4, 4)，用于对 LiDAR 点云进行旋转、缩放等操作
    :return: 过滤后的图像平面上的点，大小为 [N, 2]，表示像素坐标
    """
    
    # 1. 将点云扩展为齐次坐标 (X, Y, Z, 1)，以便与 4x4 矩阵相乘
    points_homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)  # [N, 4]

    # 2. 如果提供了 lidar_aug_matrix，对 LiDAR 点云应用增强变换（缩放、旋转等）
    if lidar_aug_matrix is not None:
        points_homogeneous = torch.matmul(lidar_aug_matrix, points_homogeneous.T).T  # [N, 4]

    # 3. 使用 lidar2camera 矩阵将 LiDAR 坐标系下的点转换为相机坐标系下的点
    points_camera = torch.matmul(lidar2camera, points_homogeneous.T).T  # [N, 4]
    
    # 4. 从相机坐标系中提取 X, Y, Z 坐标（齐次坐标，第四列被丢弃）
    points_camera = points_camera[:, :3]  # [N, 3]

    # 5. 过滤掉深度 Z_c 小于等于 0 的点
    valid_depth_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_depth_mask]  # 只保留深度为正的点
    
    # 6. 将相机坐标系下的点归一化 (X_c / Z_c, Y_c / Z_c)，并添加一个 1，形成齐次坐标
    points_camera_normalized = torch.cat([points_camera[:, :2] / points_camera[:, 2:3], torch.ones((points_camera.shape[0], 1), device=points.device)], dim=1)  # [N, 3]
    
    # 7. 使用相机内参矩阵将归一化坐标转换为图像上的像素坐标 (N, 4)
    points_image_homogeneous = torch.matmul(camera_intrinsics, torch.cat([points_camera_normalized, torch.ones((points_camera_normalized.shape[0], 1), device=points.device)], dim=1).T).T  # [N, 4]

    # 8. 应用图像增强矩阵进行变换
    points_image_augmented = torch.matmul(image_aug_matrix, points_image_homogeneous.T).T  # [N, 4]
    
    # 9. 执行透视除法，将齐次坐标转换为 2D 像素坐标
    points_image = points_image_augmented[:, :2] / points_image_augmented[:, 3:4]  # [N, 2]
    
    # 10. 过滤掉不在图像范围内的点
    valid_x_mask = (points_image[:, 0] >= 0) & (points_image[:, 0] < img_width)  # 保留在 [0, img_width) 范围内的 u 坐标
    valid_y_mask = (points_image[:, 1] >= 0) & (points_image[:, 1] < img_height)  # 保留在 [0, img_height) 范围内的 v 坐标
    valid_mask = valid_x_mask & valid_y_mask  # 同时满足 u 和 v 坐标范围的点
    
    # 11. 应用过滤条件，得到在图像范围内的有效点
    points_image_filtered = points_image[valid_mask]
    points_depth_filtered = points_camera[valid_mask][:, 2]  # 保留对应的深度信息
    
    # 返回过滤后的图像上的 2D 像素坐标和深度信息
    return points_image_filtered, points_depth_filtered