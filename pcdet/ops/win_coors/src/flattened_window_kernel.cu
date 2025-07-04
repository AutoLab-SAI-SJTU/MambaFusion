#include <torch/extension.h>
#include <cublas_v2.h>

using namespace at;

__global__ void get_window_coors_shift_v2_cuda_kernel(
    const int64_t* __restrict__ coords,
    const int64_t num_coords,
    const int64_t sparse_shape_x,
    const int64_t sparse_shape_y,
    const int64_t sparse_shape_z,
    const int64_t win_shape_x,
    const int64_t win_shape_y,
    const int64_t win_shape_z,
    const bool shift,
    int64_t* __restrict__ batch_win_inds_x,
    int64_t* __restrict__ batch_win_inds_y,
    int64_t* __restrict__ coors_in_win  // Shape: [num_coords, 3]
) {
    const int64_t shift_x = shift ? win_shape_x / 2 : 0;
    const int64_t shift_y = shift ? win_shape_y / 2 : 0;
    const int64_t shift_z = shift ? win_shape_z / 2 : 0;

    const int64_t max_num_win_x = int64_t(ceilf(float(sparse_shape_x) / win_shape_x)) + 1;
    const int64_t max_num_win_y = int64_t(ceilf(float(sparse_shape_y) / win_shape_y)) + 1;
    const int64_t max_num_win_z = int64_t(ceilf(float(sparse_shape_z) / win_shape_z)) + 1;
    const int64_t max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_coords) return;

    int64_t batch_idx = coords[index * 4 + 0];
    int64_t z = coords[index * 4 + 1] + shift_z;
    int64_t y = coords[index * 4 + 2] + shift_y;
    int64_t x = coords[index * 4 + 3] + shift_x;

    int64_t win_coors_x = x / win_shape_x;
    int64_t win_coors_y = y / win_shape_y;
    int64_t win_coors_z = z / win_shape_z;

    int64_t coor_in_win_x = x % win_shape_x;
    int64_t coor_in_win_y = y % win_shape_y;
    int64_t coor_in_win_z = z % win_shape_z;

    int64_t batch_win_ind_x = batch_idx * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z +
                              win_coors_y * max_num_win_z + win_coors_z;

    int64_t batch_win_ind_y = batch_idx * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z +
                              win_coors_x * max_num_win_z + win_coors_z;

    batch_win_inds_x[index] = batch_win_ind_x;
    batch_win_inds_y[index] = batch_win_ind_y;

    // 将 coors_in_win_z, coors_in_win_y, coors_in_win_x 写入 coors_in_win
    coors_in_win[index * 3 + 0] = coor_in_win_z;
    coors_in_win[index * 3 + 1] = coor_in_win_y;
    coors_in_win[index * 3 + 2] = coor_in_win_x;
}


std::vector<Tensor> get_window_coors_shift_v2_cuda(
    Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
) {
    const auto num_coords = coords.size(0);

    // 将 coords 转换为 int64 类型
    coords = coords.to(kLong);

    // 分配输出张量
    auto batch_win_inds_x = torch::empty({num_coords}, coords.options());
    auto batch_win_inds_y = torch::empty({num_coords}, coords.options());
    auto coors_in_win = torch::empty({num_coords, 3}, coords.options());  // Shape: [num_coords, 3]

    const int threads = 256;
    const int blocks = (num_coords + threads - 1) / threads;

    // 启动 CUDA 内核
    get_window_coors_shift_v2_cuda_kernel<<<blocks, threads>>>(
        coords.data_ptr<int64_t>(),
        num_coords,
        sparse_shape[2], // x
        sparse_shape[1], // y
        sparse_shape[0], // z
        window_shape[0], // x
        window_shape[1], // y
        window_shape[2], // z
        shift,
        batch_win_inds_x.data_ptr<int64_t>(),
        batch_win_inds_y.data_ptr<int64_t>(),
        coors_in_win.data_ptr<int64_t>()  // 传递 coors_in_win 指针
    );

    return {batch_win_inds_x, batch_win_inds_y, coors_in_win};
}

__global__ void flattened_window_mapping_kernel(
    const int64_t* __restrict__ num_per_batch,
    const int64_t* __restrict__ num_per_batch_p,
    const int64_t* __restrict__ batch_start_indices,
    const int64_t* __restrict__ batch_start_indices_p,
    int64_t* __restrict__ flat2win,
    int64_t* __restrict__ win2flat,
    int64_t group_size,
    int64_t batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size)
        return;

    int64_t num_per_batch_i = num_per_batch[i];
    int64_t num_per_batch_p_i = num_per_batch_p[i];
    int64_t batch_start_idx = batch_start_indices[i];
    int64_t batch_start_idx_p = batch_start_indices_p[i];
    int64_t batch_end_idx_p = batch_start_indices_p[i + 1];
    int64_t batch_end_idx = batch_start_indices[i + 1];
    int64_t bias_index = batch_start_idx_p - batch_start_idx;

    if (num_per_batch_i != num_per_batch_p_i) {
        int64_t remainder = num_per_batch_i % group_size;
        int64_t flat_length = batch_end_idx_p - batch_start_idx_p;

        int64_t start_p = batch_end_idx_p - group_size + remainder;
        int64_t end_p = batch_end_idx_p;

        int64_t start_p2 = batch_end_idx_p - 2 * group_size + remainder;
        int64_t end_p2 = batch_end_idx_p - group_size;

        if ((flat_length - group_size) != 0) {
            // 将 flat2win[start_p2:end_p2] 复制到 flat2win[start_p:end_p]
            for (int64_t idx = start_p; idx < end_p; ++idx) {
                flat2win[idx] = flat2win[idx - group_size];
            }
        } else {
            int64_t repeat_times = flat_length / num_per_batch_i + 1;
            int64_t temp_size = group_size - remainder;

            for (int64_t idx = 0; idx < temp_size; ++idx) {
                int64_t win2flat_idx = batch_start_idx + (idx % num_per_batch_i);
                flat2win[start_p + idx] = win2flat[win2flat_idx] + bias_index;
            }
        }
    }

    // 更新 win2flat[batch_start_idx:batch_end_idx]
    int64_t bias = batch_start_idx_p - batch_start_idx;
    for (int64_t idx = batch_start_idx; idx < batch_end_idx; ++idx) {
        win2flat[idx] += bias;
    }

    // 更新 flat2win[batch_start_idx_p:batch_end_idx_p]
    for (int64_t idx = batch_start_idx_p; idx < batch_end_idx_p; ++idx) {
        flat2win[idx] -= bias;
    }
}

std::vector<Tensor> flattened_window_mapping_cuda(
    Tensor num_per_batch,
    Tensor num_per_batch_p,
    Tensor batch_start_indices,
    Tensor batch_start_indices_p,
    int64_t group_size,
    int64_t batch_size
) {
    // 确保张量是 int64 类型并在 CUDA 上
    num_per_batch = num_per_batch.to(kLong).contiguous();
    num_per_batch_p = num_per_batch_p.to(kLong).contiguous();
    batch_start_indices = batch_start_indices.to(kLong).contiguous();
    batch_start_indices_p = batch_start_indices_p.to(kLong).contiguous();

    // 分配 flat2win 和 win2flat
    int64_t flat2win_size = batch_start_indices_p[batch_size].item<int64_t>();
    int64_t win2flat_size = batch_start_indices[batch_size].item<int64_t>();

    auto options = num_per_batch.options();

    Tensor flat2win = at::arange(flat2win_size, options);
    Tensor win2flat = at::arange(win2flat_size, options);

    // 将 flat2win 和 win2flat 移动到 CUDA 上
    flat2win = flat2win.to(kCUDA);
    win2flat = win2flat.to(kCUDA);

    // CUDA 内核配置
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // 启动 CUDA 内核
    flattened_window_mapping_kernel<<<blocks, threads>>>(
        num_per_batch.data_ptr<int64_t>(),
        num_per_batch_p.data_ptr<int64_t>(),
        batch_start_indices.data_ptr<int64_t>(),
        batch_start_indices_p.data_ptr<int64_t>(),
        flat2win.data_ptr<int64_t>(),
        win2flat.data_ptr<int64_t>(),
        group_size,
        batch_size
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in flattened_window_mapping_kernel: %s\n", cudaGetErrorString(err));
    }

    return {win2flat, flat2win};
}

__global__ void get_window_coors_shift_v3_cuda_kernel(
    const int64_t* __restrict__ coords,
    const int64_t num_coords,
    const int64_t sparse_shape_x,
    const int64_t sparse_shape_y,
    const int64_t sparse_shape_z,
    const int64_t win_shape_x,
    const int64_t win_shape_y,
    const int64_t win_shape_z,
    const bool shift,
    int64_t* __restrict__ vx,
    int64_t* __restrict__ vy
) {
    const int64_t shift_x = shift ? win_shape_x / 2 : 0;
    const int64_t shift_y = shift ? win_shape_y / 2 : 0;
    const int64_t shift_z = shift ? win_shape_z / 2 : 0;

    const int64_t max_num_win_x = int64_t(ceilf(float(sparse_shape_x) / win_shape_x)) + 1;
    const int64_t max_num_win_y = int64_t(ceilf(float(sparse_shape_y) / win_shape_y)) + 1;
    const int64_t max_num_win_z = int64_t(ceilf(float(sparse_shape_z) / win_shape_z)) + 1;
    const int64_t max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_coords) return;

    int64_t batch_idx = coords[index * 4 + 0];
    int64_t z = coords[index * 4 + 1] + shift_z;
    int64_t y = coords[index * 4 + 2] + shift_y;
    int64_t x = coords[index * 4 + 3] + shift_x;

    int64_t win_coors_x = x / win_shape_x;
    int64_t win_coors_y = y / win_shape_y;
    int64_t win_coors_z = z / win_shape_z;

    int64_t coor_in_win_x = x % win_shape_x;
    int64_t coor_in_win_y = y % win_shape_y;
    int64_t coor_in_win_z = z % win_shape_z;

    int64_t batch_win_ind_x = batch_idx * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z +
                              win_coors_y * max_num_win_z + win_coors_z;

    int64_t batch_win_ind_y = batch_idx * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z +
                              win_coors_x * max_num_win_z + win_coors_z;

    // 计算 vx
    int64_t temp1 = win_shape_x * win_shape_y * win_shape_z;
    int64_t temp2_x = batch_win_ind_x * temp1;
    int64_t temp3_x = coor_in_win_x * win_shape_y * win_shape_z +
                      coor_in_win_y * win_shape_z +
                      coor_in_win_z;
    vx[index] = temp2_x + temp3_x;

    // 计算 vy
    int64_t temp2_y = batch_win_ind_y * temp1;
    int64_t temp3_y = coor_in_win_y * win_shape_x * win_shape_z +
                      coor_in_win_x * win_shape_z +
                      coor_in_win_z;
    vy[index] = temp2_y + temp3_y;
}

std::vector<Tensor> get_window_coors_shift_v3_cuda(
    Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
) {
    const auto num_coords = coords.size(0);

    // 将 coords 转换为 int64 类型
    coords = coords.to(kLong);

    // 分配输出张量
    auto vx = torch::empty({num_coords}, coords.options());
    auto vy = torch::empty({num_coords}, coords.options());

    const int threads = 256;
    const int blocks = (num_coords + threads - 1) / threads;

    // 启动 CUDA 内核
    get_window_coors_shift_v3_cuda_kernel<<<blocks, threads>>>(
        coords.data_ptr<int64_t>(),
        num_coords,
        sparse_shape[2], // x
        sparse_shape[1], // y
        sparse_shape[0], // z
        window_shape[0], // x
        window_shape[1], // y
        window_shape[2], // z
        shift,
        vx.data_ptr<int64_t>(),
        vy.data_ptr<int64_t>()
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in get_window_coors_shift_v3_cuda_kernel: %s\n", cudaGetErrorString(err));
    }

    return {vx, vy};
}


// CUDA 内核函数
__global__ void expand_selected_coords_cuda_kernel(
    const int* __restrict__ selected_coords_copy,  // [selected_coords_num, 4]
    int* __restrict__ selected_coords_expand,      // [selected_coords_num * diffusion_scale, 4]
    const int selected_coords_num,
    const int diffusion_scale,
    const int coords_shift,
    const int h,
    const int w,
    const int d
) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = selected_coords_num * diffusion_scale;
    if (idx >= total) return;

    // 计算对应的原始坐标索引和扩散索引
    int coord_idx = idx % selected_coords_num;    // 原始坐标索引
    int scale_idx = idx / selected_coords_num;    // 扩散索引 (0 到 diffusion_scale - 1)

    // 读取原始坐标
    int batch_idx = selected_coords_copy[coord_idx * 4 + 0];
    int z = selected_coords_copy[coord_idx * 4 + 1];
    int y = selected_coords_copy[coord_idx * 4 + 2];
    int x = selected_coords_copy[coord_idx * 4 + 3];

    // 根据 scale_idx 确定偏移后的坐标
    int new_z = z;
    int new_y = y;
    int new_x = x;

    if (diffusion_scale == 2) {
        if (scale_idx == 0) {
            // 第一次扩散
            new_x = max(min(x - coords_shift, w - 1), 0);  // 确保 x、w 使用 int
            new_y = max(min(y + coords_shift, h - 1), 0);  // 确保 y、h 使用 int
            new_z = max(min(z, d - 1), 0);                 // 确保 z、d 使用 int
        } else if (scale_idx == 1) {
            // 第二次扩散
            new_x = max(min(x + coords_shift, w - 1), 0);
            new_y = max(min(y + coords_shift, h - 1), 0);
            new_z = max(min(z, d - 1), 0);
        }
    } else if (diffusion_scale == 4) {
        if (scale_idx == 0) {
            new_x = max(min(x - coords_shift, w - 1), 0);
            new_y = max(min(y + coords_shift, h - 1), 0);
            new_z = max(min(z, d - 1), 0);
        } else if (scale_idx == 1) {
            new_x = max(min(x + coords_shift, w - 1), 0);
            new_y = max(min(y + coords_shift, h - 1), 0);
            new_z = max(min(z, d - 1), 0);
        } else if (scale_idx == 2) {
            new_x = max(min(x - coords_shift, w - 1), 0);
            new_y = max(min(y - coords_shift, h - 1), 0);
            new_z = max(min(z, d - 1), 0);
        } else if (scale_idx == 3) {
            new_x = max(min(x + coords_shift, w - 1), 0);
            new_y = max(min(y - coords_shift, h - 1), 0);
            new_z = max(min(z, d - 1), 0);
        }
    }

    // 将新的坐标写入 selected_coords_expand
    selected_coords_expand[idx * 4 + 0] = batch_idx;
    selected_coords_expand[idx * 4 + 1] = new_z;
    selected_coords_expand[idx * 4 + 2] = new_y;
    selected_coords_expand[idx * 4 + 3] = new_x;
}

// CUDA 实现函数
std::vector<at::Tensor> expand_selected_coords_cuda(
    at::Tensor selected_coords_copy,  // [selected_coords_num, 4]
    int diffusion_scale,
    int coords_shift,
    int h,
    int w,
    int d,
    int C  // 特征维度
) {
    // 获取输入的尺寸
    int selected_coords_num = selected_coords_copy.size(0);

    // 分配输出张量
    auto options_coords = selected_coords_copy.options();
    auto selected_coords_expand = at::empty({selected_coords_num * diffusion_scale, 4}, options_coords);

    // 初始化特征张量为全零
    auto options_feats = at::device(selected_coords_copy.device()).dtype(at::kFloat);
    auto selected_feats_expand = at::zeros({selected_coords_num * diffusion_scale, C}, options_feats);

    // 定义 CUDA 内核配置
    int threads = 256;
    int blocks = (selected_coords_num * diffusion_scale + threads - 1) / threads;

    // 启动 CUDA 内核
    expand_selected_coords_cuda_kernel<<<blocks, threads>>>(
        selected_coords_copy.data_ptr<int>(),
        selected_coords_expand.data_ptr<int>(),
        selected_coords_num,
        diffusion_scale,
        coords_shift,
        h,
        w,
        d
    );

    return {selected_coords_expand, selected_feats_expand};
}

__global__ void map_points_cuda_kernel(
    const float* __restrict__ points,          // [grid_num, sample_num, 3]
    const float* __restrict__ lidar2image,     // [batch_size, num_view, 4, 4]
    const float* __restrict__ image_aug_matrix,// [batch_size, num_view, 4, 4]
    float* __restrict__ points_out,            // [batch_size, num_view, grid_num, sample_num, 4, 1]
    float* __restrict__ points_2d_out,         // [batch_size, num_view, grid_num, sample_num, 2]
    bool* __restrict__ map_mask_out,           // [batch_size, num_view, grid_num, sample_num, 1]
    // float* __restrict__ point_camera_out,      // [batch_size, num_view, grid_num, sample_num, 4]
    // float* __restrict__ point_augmented_out,   // [batch_size, num_view, grid_num, sample_num, 4]
    int batch_size,
    int num_view,
    int grid_num,
    int sample_num,
    int image_height,
    int image_width,
    float expand_scale
) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_view * grid_num * sample_num;
    if (idx >= total_points) return;

    // 计算batch、view、grid和sample的索引
    int sample_idx = idx % sample_num;
    int grid_idx = (idx / sample_num) % grid_num;
    int view_idx = (idx / (sample_num * grid_num)) % num_view;
    int batch_idx = (idx / (sample_num * grid_num * num_view)) % batch_size;

    // 读取3D点并添加齐次坐标
    int point_idx = ((grid_idx * sample_num) + sample_idx) * 3;
    float point[4];  // 齐次坐标
    for (int i = 0; i < 3; ++i) {
        point[i] = points[point_idx + i];
    }
    point[3] = 1.0f;  // 添加齐次坐标

    // 读取当前batch和view的lidar2image矩阵
    int lidar2image_idx = (((batch_idx * num_view) + view_idx) * 16);
    float lidar2image_mat[4][4];
    for (int i = 0; i < 16; ++i) {
        lidar2image_mat[i / 4][i % 4] = lidar2image[lidar2image_idx + i];
    }

    // 进行坐标变换
    float point_camera[4] = {0.0f};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            point_camera[i] += lidar2image_mat[i][j] * point[j];
        }
    }

    // 判断点是否在相机前方
    float eps = 1e-5f;
    bool mask = point_camera[2] > eps;

    // 进行透视投影
    float z = fmaxf(point_camera[2], eps);
    float x = point_camera[0] / z;
    float y = point_camera[1] / z;

    // 设置齐次坐标
    // point_camera[2] = 1.0f;

    // 读取当前batch和view的image_aug_matrix
    int image_aug_matrix_idx = (((batch_idx * num_view) + view_idx) * 16);
    float image_aug_mat[4][4];
    for (int i = 0; i < 16; ++i) {
        image_aug_mat[i / 4][i % 4] = image_aug_matrix[image_aug_matrix_idx + i];
    }

    // 创建图像坐标下的点
    float point_image[4] = { x, y, 1.0f, 1.0f };

    // 应用图像增强变换
    float point_augmented[4] = {0.0f};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            point_augmented[i] += image_aug_mat[i][j] * point_image[j];
        }
    }

    // 获取增强后的2D坐标
    float u = point_augmented[0];
    float v = point_augmented[1];

    // 归一化坐标
    u /= static_cast<float>(image_width);
    v /= static_cast<float>(image_height);

    // 更新mask，判断点是否在图像范围内
    mask = mask && (v > -expand_scale) && (v < 1.0f + expand_scale)
                    && (u > -expand_scale) && (u < 1.0f + expand_scale);

    // 写入输出数据
    int points_out_idx = ((((((batch_idx * num_view + view_idx) * grid_num + grid_idx) * sample_num) + sample_idx) * 4) * 1);
    int points_2d_out_idx = (((((batch_idx * num_view + view_idx) * grid_num + grid_idx) * sample_num) + sample_idx) * 2);
    int map_mask_out_idx = (((((batch_idx * num_view + view_idx) * grid_num + grid_idx) * sample_num) + sample_idx) * 1);

    // 写入points_out
    for (int i = 0; i < 4; ++i) {
        points_out[points_out_idx + i * 1] = point[i]; // 形状：[4, 1]
    }

    // 写入points_2d_out
    points_2d_out[points_2d_out_idx + 0] = u;
    points_2d_out[points_2d_out_idx + 1] = v;

    // 写入map_mask_out
    map_mask_out[map_mask_out_idx] = mask;

    // 写入 point_camera 和 point_augmented
    // int point_camera_out_idx = points_out_idx;
    // int point_augmented_out_idx = points_out_idx;
    // for (int i = 0; i < 4; ++i) {
    //     point_camera_out[point_camera_out_idx + i] = point_camera[i];
    //     point_augmented_out[point_augmented_out_idx + i] = point_augmented[i];
    // }
}

// CUDA implementation function
// std::vector<at::Tensor> map_points_cuda(
//     at::Tensor points,
//     at::Tensor lidar2image,
//     at::Tensor image_aug_matrix,
//     int batch_size,
//     int height,
//     int width,
//     float expand_scale
// ) {
//     // Get sizes
//     int grid_num = points.size(0);
//     int sample_num = points.size(1);
//     int num_view = lidar2image.size(1);

//     // Total number of points
//     int total_points = batch_size * num_view * grid_num * sample_num;

//     // Allocate output tensors
//     auto options = points.options();
//     at::Tensor points_out = at::empty({batch_size, num_view, grid_num, sample_num, 4}, options);
//     at::Tensor points_2d_out = at::empty({batch_size, num_view, grid_num, sample_num, 2}, options);
//     at::Tensor map_mask_out = at::empty({batch_size, num_view, grid_num, sample_num}, options.dtype(at::kBool));

//     // Define CUDA kernel configuration
//     int threads = 256;
//     int blocks = (total_points + threads - 1) / threads;

//     // Launch CUDA kernel
//     map_points_cuda_kernel<<<blocks, threads>>>(
//         points.data_ptr<float>(),
//         lidar2image.data_ptr<float>(),
//         image_aug_matrix.data_ptr<float>(),
//         points_out.data_ptr<float>(),
//         points_2d_out.data_ptr<float>(),
//         map_mask_out.data_ptr<bool>(),
//         batch_size,
//         num_view,
//         grid_num,
//         sample_num,
//         height,
//         width,
//         expand_scale
//     );

//     return {points_out, points_2d_out, map_mask_out};
// }

std::vector<at::Tensor> map_points_cuda(
    at::Tensor points,
    at::Tensor lidar2image,
    at::Tensor image_aug_matrix,
    int batch_size,
    int height,
    int width,
    float expand_scale
) {
    int grid_num = points.size(0);
    int sample_num = points.size(1);
    int num_view = lidar2image.size(1);
    int total_points = batch_size * num_view * grid_num * sample_num;

    auto options = points.options();
    at::Tensor points_out = at::empty({batch_size, num_view, grid_num, sample_num, 4}, options);
    at::Tensor points_2d_out = at::empty({batch_size, num_view, grid_num, sample_num, 2}, options);
    at::Tensor map_mask_out = at::empty({batch_size, num_view, grid_num, sample_num}, options.dtype(at::kBool));
    // at::Tensor point_camera_out = at::empty({batch_size, num_view, grid_num, sample_num, 4}, options);
    // at::Tensor point_augmented_out = at::empty({batch_size, num_view, grid_num, sample_num, 4}, options);

    int threads = 256;
    int blocks = (total_points + threads - 1) / threads;

    map_points_cuda_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        lidar2image.data_ptr<float>(),
        image_aug_matrix.data_ptr<float>(),
        points_out.data_ptr<float>(),
        points_2d_out.data_ptr<float>(),
        map_mask_out.data_ptr<bool>(),
        // point_camera_out.data_ptr<float>(),
        // point_augmented_out.data_ptr<float>(),
        batch_size,
        num_view,
        grid_num,
        sample_num,
        height,
        width,
        expand_scale
    );

    // return {points_out, points_2d_out, map_mask_out, point_camera_out, point_augmented_out};
    return {points_out, points_2d_out, map_mask_out};
}

__global__ void process_hit_points_kernel(
    const float* __restrict__ points_2d,   // [batch_size, num_view, grid_num, sample_num, 2]
    const bool* __restrict__ map_mask,     // [batch_size, num_view, grid_num, sample_num, 1]
    float* __restrict__ hit_points_out,    // [grid_num, sample_num, 3]
    int batch_size,
    int num_view,
    int grid_num,
    int sample_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_num * sample_num) return;

    int grid_idx = idx / sample_num;
    int sample_idx = idx % sample_num;

    bool hit_found = false;
    int first_hit_view = -1;
    float hit_u = 0.0f;
    float hit_v = 0.0f;

    // 假设batch_size为1
    int batch_idx = 0;

    // 遍历所有视角，寻找第一个命中的视角
    for (int view_idx = 0; view_idx < num_view; ++view_idx) {
        int mask_idx = ((((((batch_idx * num_view) + view_idx) * grid_num + grid_idx) * sample_num) + sample_idx) * 1);
        bool mask = map_mask[mask_idx];

        if (mask) {
            if (!hit_found) {
                // 获取对应的2D坐标
                int points_2d_idx = ((((((batch_idx * num_view) + view_idx) * grid_num + grid_idx) * sample_num) + sample_idx) * 2);
                hit_u = points_2d[points_2d_idx + 0];
                hit_v = points_2d[points_2d_idx + 1];
                first_hit_view = view_idx;
            }
            hit_found = true;
            break;
        }
    }

    if (!hit_found) {
        // 如果没有命中任何视角，将视角ID设置为0
        first_hit_view = 0;
        int points_2d_idx = ((((((batch_idx * num_view) + first_hit_view) * grid_num + grid_idx) * sample_num) + sample_idx) * 2);
        hit_u = points_2d[points_2d_idx + 0];
        hit_v = points_2d[points_2d_idx + 1];
        // hit_u = 0.0f;
        // hit_v = 0.0f;
    }

    // 对坐标进行裁剪和调整
    float hit_u_clamped = fminf(fmaxf(hit_u, -1.0f), 2.0f) + 1.0f;
    float hit_v_clamped = fminf(fmaxf(hit_v, -1.0f), 2.0f) + 1.0f;

    // 写入输出张量
    int hit_points_out_idx = (grid_idx * sample_num + sample_idx) * 3;
    hit_points_out[hit_points_out_idx + 0] = hit_u_clamped;
    hit_points_out[hit_points_out_idx + 1] = hit_v_clamped;
    hit_points_out[hit_points_out_idx + 2] = static_cast<float>(first_hit_view);
}

// 添加新的CUDA实现函数，用于处理hit_points_out
std::vector<at::Tensor> map_points_v2_cuda(
    at::Tensor points,
    at::Tensor lidar2image,
    at::Tensor image_aug_matrix,
    int batch_size,
    int height,
    int width,
    float expand_scale
) {
    // 调用原有的map_points_cuda函数，获取points_2d和map_mask
    auto outputs = map_points_cuda(
        points,
        lidar2image,
        image_aug_matrix,
        batch_size,
        height,
        width,
        expand_scale
    );

    at::Tensor points_out = outputs[0];
    at::Tensor points_2d_out = outputs[1];
    at::Tensor map_mask_out = outputs[2];

    int grid_num = points.size(0);
    int sample_num = points.size(1);
    int num_view = lidar2image.size(1);

    // 分配hit_points_out张量
    auto options = points.options();
    at::Tensor hit_points_out = at::empty({grid_num, sample_num, 3}, options);

    // 定义CUDA核函数配置
    int threads = 256;
    int total_points = grid_num * sample_num;
    int blocks = (total_points + threads - 1) / threads;

    // 启动CUDA核函数
    process_hit_points_kernel<<<blocks, threads>>>(
        points_2d_out.data_ptr<float>(),
        map_mask_out.data_ptr<bool>(),
        hit_points_out.data_ptr<float>(),
        batch_size,
        num_view,
        grid_num,
        sample_num
    );

    // 返回hit_points_out
    return {hit_points_out};
}

// CUDA 实现函数
__global__ void fused_hilbert_pos_embed_kernel(
    const int64_t* __restrict__ coords_s1,    // 输入坐标 s1 [N, 4]
    // const int64_t* __restrict__ coords_s2,    // 输入坐标 s2 [N, 4]
    const int64_t* __restrict__ template_s1,  // hilbert template for s1
    // const int64_t* __restrict__ template_s2,  // hilbert template for s2
    float* __restrict__ pos_embed_s1,         // 输出位置编码 s1 [N, 9]
    // float* __restrict__ pos_embed_s2,         // 输出位置编码 s2 [N, 9]
    int64_t* __restrict__ hilbert_s1,         // 输出 hilbert 索引 s1 [N]
    // int64_t* __restrict__ hilbert_s2,         // 输出 hilbert 索引 s2 [N]
    const int64_t num_coords,                 // 坐标数量
    const int64_t hil_size_x_s1, const int64_t hil_size_y_s1, const int64_t hil_size_z_s1,  // hilbert size for s1
    // const int64_t hil_size_x_s2, const int64_t hil_size_y_s2, const int64_t hil_size_z_s2,  // hilbert size for s2
    const int64_t sparse_shape_s1_x, const int64_t sparse_shape_s1_y, const int64_t sparse_shape_s1_z,  // sparse shape s1
    // const int64_t sparse_shape_s2_x, const int64_t sparse_shape_s2_y, const int64_t sparse_shape_s2_z,  // sparse shape s2
    const int64_t shift_x, const int64_t shift_y, const int64_t shift_z          // shift values
) {
    // 获取当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_coords) return;  // 确保索引不超出范围

    // 获取当前的坐标 S1
    int64_t z_s1 = coords_s1[idx * 4 + 1] + shift_z;  // Z 轴坐标加上 shift
    int64_t y_s1 = coords_s1[idx * 4 + 2] + shift_y;  // Y 轴坐标加上 shift
    int64_t x_s1 = coords_s1[idx * 4 + 3] + shift_x;  // X 轴坐标加上 shift

    // 获取当前的坐标 S2
    // int64_t z_s2 = coords_s2[idx * 4 + 1] + shift_z;  // Z 轴坐标加上 shift
    // int64_t y_s2 = coords_s2[idx * 4 + 2] + shift_y;  // Y 轴坐标加上 shift
    // int64_t x_s2 = coords_s2[idx * 4 + 3] + shift_x;  // X 轴坐标加上 shift

    // 计算 flat_coors 并使用 hilbert template 查找索引 (S1 和 S2 分别处理)
    int64_t flat_coors_s1 = z_s1 * hil_size_y_s1 * hil_size_x_s1 + y_s1 * hil_size_x_s1 + x_s1;
    // int64_t flat_coors_s2 = z_s2 * hil_size_y_s2 * hil_size_x_s2 + y_s2 * hil_size_x_s2 + x_s2;

    // 使用 hilbert template 查找对应的 hilbert 索引
    hilbert_s1[idx] = template_s1[flat_coors_s1];
    // hilbert_s2[idx] = template_s2[flat_coors_s2];

    // 预先计算常量
    const float inv_12 = 1.0f / 12.0f;  // 1/12 只计算一次
    const float inv_sparse_shape_y_12 = 1.0f / static_cast<float>(sparse_shape_s1_y / 12 + 1);
    const float inv_sparse_shape_x_12 = 1.0f / static_cast<float>(sparse_shape_s1_x / 12 + 1);
    const float inv_sparse_shape_z = 1.0f / static_cast<float>(sparse_shape_s1_z);

    // 计算位置编码 pos_embed_s1 并避免重复除法
    pos_embed_s1[idx * 9 + 0] = z_s1 * inv_sparse_shape_z;  // Z 轴
    pos_embed_s1[idx * 9 + 1] = (y_s1 / 12) * inv_sparse_shape_y_12;  // Y 轴除以 12 并标准化
    pos_embed_s1[idx * 9 + 2] = (x_s1 / 12) * inv_sparse_shape_x_12;  // X 轴除以 12 并标准化
    pos_embed_s1[idx * 9 + 3] = (y_s1 % 12) * inv_12;  // Y 轴取余并标准化
    pos_embed_s1[idx * 9 + 4] = (x_s1 % 12) * inv_12;  // X 轴取余并标准化
    pos_embed_s1[idx * 9 + 5] = ((y_s1 + 6) / 12) * inv_sparse_shape_y_12;  // Y 轴加 6 后再标准化
    pos_embed_s1[idx * 9 + 6] = ((x_s1 + 6) / 12) * inv_sparse_shape_x_12;  // X 轴加 6 后再标准化
    pos_embed_s1[idx * 9 + 7] = ((y_s1 + 6) % 12) * inv_12;  // Y 轴加 6 后取余并标准化
    pos_embed_s1[idx * 9 + 8] = ((x_s1 + 6) % 12) * inv_12;  // X 轴加 6 后取余并标准化
    
    // --- 计算位置编码 pos_embed_s1 ---
    // pos_embed_s1[idx * 9 + 0] = coords_s1[idx * 4 + 1] / static_cast<float>(sparse_shape_s1_z);  // Z 轴
    // pos_embed_s1[idx * 9 + 1] = (coords_s1[idx * 4 + 2] / 12) / static_cast<float>(sparse_shape_s1_y / 12 + 1);  // Y 轴除以 12
    // pos_embed_s1[idx * 9 + 2] = (coords_s1[idx * 4 + 3] / 12) / static_cast<float>(sparse_shape_s1_y / 12 + 1);  // X 轴除以 12
    // pos_embed_s1[idx * 9 + 3] = (coords_s1[idx * 4 + 2] % 12) / 12.0f;  // 取余 12
    // pos_embed_s1[idx * 9 + 4] = (coords_s1[idx * 4 + 3] % 12) / 12.0f;  // 取余 12
    // pos_embed_s1[idx * 9 + 5] = ((coords_s1[idx * 4 + 2] + 6) / 12) / static_cast<float>(sparse_shape_s1_y / 12 + 1);  // Y 轴位移加 6 再除以 12
    // pos_embed_s1[idx * 9 + 6] = ((coords_s1[idx * 4 + 3] + 6) / 12) / static_cast<float>(sparse_shape_s1_y / 12 + 1);  // X 轴位移加 6 再除以 12
    // pos_embed_s1[idx * 9 + 7] = ((coords_s1[idx * 4 + 2] + 6) % 12) / 12.0f;  // 位移后取余 12
    // pos_embed_s1[idx * 9 + 8] = ((coords_s1[idx * 4 + 3] + 6) % 12) / 12.0f;  // 位移后取余 12

    // // --- 计算位置编码 pos_embed_s2 ---
    // pos_embed_s2[idx * 9 + 0] = coords_s2[idx * 4 + 1] / static_cast<float>(sparse_shape_s1_z);  // Z 轴
    // pos_embed_s2[idx * 9 + 1] = (coords_s2[idx * 4 + 2] / 12) / static_cast<float>(sparse_shape_s2_y / 12 + 1);  // Y 轴除以 12
    // pos_embed_s2[idx * 9 + 2] = (coords_s2[idx * 4 + 3] / 12) / static_cast<float>(sparse_shape_s2_y / 12 + 1);  // X 轴除以 12
    // pos_embed_s2[idx * 9 + 3] = (coords_s2[idx * 4 + 2] % 12) / 12.0f;  // 取余 12
    // pos_embed_s2[idx * 9 + 4] = (coords_s2[idx * 4 + 3] % 12) / 12.0f;  // 取余 12
    // pos_embed_s2[idx * 9 + 5] = ((coords_s2[idx * 4 + 2] + 6) / 12) / static_cast<float>(sparse_shape_s2_y / 12 + 1);  // Y 轴位移加 6 再除以 12
    // pos_embed_s2[idx * 9 + 6] = ((coords_s2[idx * 4 + 3] + 6) / 12) / static_cast<float>(sparse_shape_s2_y / 12 + 1);  // X 轴位移加 6 再除以 12
    // pos_embed_s2[idx * 9 + 7] = ((coords_s2[idx * 4 + 2] + 6) % 12) / 12.0f;  // 位移后取余 12
    // pos_embed_s2[idx * 9 + 8] = ((coords_s2[idx * 4 + 3] + 6) % 12) / 12.0f;  // 位移后取余 12
}



// CUDA 实现函数
std::vector<at::Tensor> fused_hilbert_pos_embed_cuda(
    at::Tensor coords_s1,                    // 输入坐标 s1 [N, 4]
    // at::Tensor coords_s2,                    // 输入坐标 s2 [N, 4]
    at::Tensor template_s1,                  // hilbert template for s1
    // at::Tensor template_s2,                  // hilbert template for s2
    int64_t batch_size,                      // batch size
    int64_t hil_size_z_s1, int64_t hil_size_y_s1, int64_t hil_size_x_s1,  // hilbert sizes for s1
    // int64_t hil_size_z_s2, int64_t hil_size_y_s2, int64_t hil_size_x_s2,  // hilbert sizes for s2
    std::vector<int64_t> sparse_shape_s1,    // sparse shape for s1
    // std::vector<int64_t> sparse_shape_s2,    // sparse shape for s2
    std::vector<int64_t> shift               // shift values for z, y, x
) {
    const int64_t num_coords = coords_s1.size(0);  // 坐标数量

    // 将 shift 的 z, y, x 值从向量中提取出来
    int64_t shift_z = shift[0];
    int64_t shift_y = shift[1];
    int64_t shift_x = shift[2];

    // 分配输出张量
    auto hilbert_s1 = torch::empty({num_coords}, coords_s1.options());  // Hilbert 索引 s1
    // auto hilbert_s2 = torch::empty({num_coords}, coords_s2.options());  // Hilbert 索引 s2
    auto pos_embed_s1 = torch::empty({num_coords, 9}, coords_s1.options().dtype(at::kFloat));  // 位置编码 s1
    // auto pos_embed_s2 = torch::empty({num_coords, 9}, coords_s2.options().dtype(at::kFloat));  // 位置编码 s2

    // 定义 CUDA 内核的配置
    const int threads = 256;  // 每个 block 中的线程数
    const int blocks = (num_coords + threads - 1) / threads;  // 计算 block 数量

    // 启动 CUDA 内核，执行并行计算
    fused_hilbert_pos_embed_kernel<<<blocks, threads>>>(
        coords_s1.data_ptr<int64_t>(),        // s1 的坐标数据指针
        // coords_s2.data_ptr<int64_t>(),        // s2 的坐标数据指针
        template_s1.data_ptr<int64_t>(),      // s1 的 Hilbert template
        // template_s2.data_ptr<int64_t>(),      // s2 的 Hilbert template
        pos_embed_s1.data_ptr<float>(),       // s1 的位置编码输出
        // pos_embed_s2.data_ptr<float>(),       // s2 的位置编码输出
        hilbert_s1.data_ptr<int64_t>(),       // s1 的 Hilbert 索引输出
        // hilbert_s2.data_ptr<int64_t>(),       // s2 的 Hilbert 索引输出
        num_coords,                           // 坐标数量
        hil_size_x_s1, hil_size_y_s1, hil_size_z_s1,   // Hilbert size for s1
        // hil_size_x_s2, hil_size_y_s2, hil_size_z_s2,   // Hilbert size for s2
        sparse_shape_s1[2], sparse_shape_s1[1], sparse_shape_s1[0],  // s1 的 sparse shape
        // sparse_shape_s2[2], sparse_shape_s2[1], sparse_shape_s2[0],  // s2 的 sparse shape
        shift_x, shift_y, shift_z             // shift 的 x, y, z 维度
    );

    // 返回结果，包括 Hilbert 索引和位置编码
    // return {hilbert_s1, hilbert_s2, pos_embed_s1, pos_embed_s2};
    return {hilbert_s1, pos_embed_s1};
}
