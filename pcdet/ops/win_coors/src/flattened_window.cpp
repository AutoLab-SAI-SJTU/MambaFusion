#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

// 声明 CUDA 函数
std::vector<at::Tensor> get_window_coors_shift_v2_cuda(
    at::Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
);

// C++ 接口（暴露给 Python）
std::vector<at::Tensor> get_window_coors_shift_v2(
    at::Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
) {
    // 检查输入
    TORCH_CHECK(coords.is_cuda(), "coords 必须是 CUDA 张量");
    return get_window_coors_shift_v2_cuda(coords, sparse_shape, window_shape, shift);
}

// 声明 CUDA 函数
std::vector<at::Tensor> flattened_window_mapping_cuda(
    at::Tensor num_per_batch,
    at::Tensor num_per_batch_p,
    at::Tensor batch_start_indices,
    at::Tensor batch_start_indices_p,
    int64_t group_size,
    int64_t batch_size
);

// C++ 接口（暴露给 Python）
std::vector<at::Tensor> flattened_window_mapping(
    at::Tensor num_per_batch,
    at::Tensor num_per_batch_p,
    at::Tensor batch_start_indices,
    at::Tensor batch_start_indices_p,
    int64_t group_size,
    int64_t batch_size
) {
    // 检查输入
    TORCH_CHECK(num_per_batch.is_cuda(), "num_per_batch 必须是 CUDA 张量");
    TORCH_CHECK(num_per_batch_p.is_cuda(), "num_per_batch_p 必须是 CUDA 张量");
    TORCH_CHECK(batch_start_indices.is_cuda(), "batch_start_indices 必须是 CUDA 张量");
    TORCH_CHECK(batch_start_indices_p.is_cuda(), "batch_start_indices_p 必须是 CUDA 张量");

    return flattened_window_mapping_cuda(
        num_per_batch,
        num_per_batch_p,
        batch_start_indices,
        batch_start_indices_p,
        group_size,
        batch_size
    );
}




// 声明 CUDA 函数
std::vector<at::Tensor> get_window_coors_shift_v3_cuda(
    at::Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
);

// C++ 接口（暴露给 Python）
std::vector<at::Tensor> get_window_coors_shift_v3(
    at::Tensor coords,
    std::vector<int64_t> sparse_shape,
    std::vector<int64_t> window_shape,
    bool shift
) {
    // 检查输入
    TORCH_CHECK(coords.is_cuda(), "coords 必须是 CUDA 张量");
    return get_window_coors_shift_v3_cuda(coords, sparse_shape, window_shape, shift);
}


// 声明 CUDA 函数
std::vector<at::Tensor> expand_selected_coords_cuda(
    at::Tensor selected_coords_copy,  // [selected_coords_num, 4]
    int diffusion_scale,
    int coords_shift,
    int h,
    int w,
    int d,
    int C  // 特征维度
);

// C++ 接口（暴露给 Python）
std::vector<at::Tensor> expand_selected_coords(
    at::Tensor selected_coords_copy,  // [selected_coords_num, 4]
    int diffusion_scale,
    int coords_shift,
    int h,
    int w,
    int d,
    int C  // 特征维度
) {
    // 检查输入张量是否在 CUDA 上
    TORCH_CHECK(selected_coords_copy.is_cuda(), "selected_coords_copy 必须是 CUDA 张量");

    return expand_selected_coords_cuda(
        selected_coords_copy,
        diffusion_scale,
        coords_shift,
        h,
        w,
        d,
        C
    );
}

// Declaration of CUDA function
std::vector<at::Tensor> map_points_cuda(
    at::Tensor points,              // [grid_num, sample_num, 4]
    at::Tensor lidar2image,         // [batch_size, num_view, 4, 4]
    at::Tensor image_aug_matrix,    // [batch_size, num_view, 4, 4]
    int batch_size,
    int height,
    int width,
    float expand_scale
);

// C++ interface (exposed to Python)
std::vector<at::Tensor> map_points(
    at::Tensor points,
    at::Tensor lidar2image,
    at::Tensor image_aug_matrix,
    int batch_size,
    int height,
    int width,
    float expand_scale = 0.0
) {
    // Check if inputs are on CUDA
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(lidar2image.is_cuda(), "lidar2image must be a CUDA tensor");
    TORCH_CHECK(image_aug_matrix.is_cuda(), "image_aug_matrix must be a CUDA tensor");

    return map_points_cuda(
        points,
        lidar2image,
        image_aug_matrix,
        batch_size,
        height,
        width,
        expand_scale
    );
}

// Declaration of the new CUDA function
std::vector<at::Tensor> map_points_v2_cuda(
    at::Tensor points,              // [grid_num, 3]
    at::Tensor lidar2image,         // [batch_size, num_view, 4, 4]
    at::Tensor image_aug_matrix,    // [batch_size, num_view, 4, 4]
    int batch_size,
    int height,
    int width,
    float expand_scale
);

// C++ interface (exposed to Python)
std::vector<at::Tensor> map_points_v2(
    at::Tensor points,
    at::Tensor lidar2image,
    at::Tensor image_aug_matrix,
    int batch_size,
    int height,
    int width,
    float expand_scale = 0.0
) {
    // Check if inputs are on CUDA
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(lidar2image.is_cuda(), "lidar2image must be a CUDA tensor");
    TORCH_CHECK(image_aug_matrix.is_cuda(), "image_aug_matrix must be a CUDA tensor");

    return map_points_v2_cuda(
        points,
        lidar2image,
        image_aug_matrix,
        batch_size,
        height,
        width,
        expand_scale
    );
}

// 声明 fused_hilbert_pos_embed_cuda 函数
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
);



// Python 接口，用于从 Python 中调用 C++ 函数
std::vector<at::Tensor> fused_hilbert_pos_embed(
    at::Tensor coords_s1,                   // 输入坐标 s1 [N, 4]
    // at::Tensor coords_s2,                   // 输入坐标 s2 [N, 4]
    at::Tensor template_s1,                 // hilbert template for s1
    // at::Tensor template_s2,                 // hilbert template for s2
    int64_t batch_size,                     // batch size
    int64_t hil_size_z_s1, int64_t hil_size_y_s1, int64_t hil_size_x_s1,  // hilbert sizes for s1
    // int64_t hil_size_z_s2, int64_t hil_size_y_s2, int64_t hil_size_x_s2,  // hilbert sizes for s2
    std::vector<int64_t> sparse_shape_s1,   // sparse shape for s1
    // std::vector<int64_t> sparse_shape_s2,   // sparse shape for s2
    std::vector<int64_t> shift              // shift values for z, y, x
) {
    // 确保输入张量是 CUDA 张量
    TORCH_CHECK(coords_s1.is_cuda(), "coords_s1 必须是 CUDA 张量");
    // TORCH_CHECK(coords_s2.is_cuda(), "coords_s2 必须是 CUDA 张量");
    TORCH_CHECK(template_s1.is_cuda(), "template_s1 必须是 CUDA 张量");
    // TORCH_CHECK(template_s2.is_cuda(), "template_s2 必须是 CUDA 张量");

    // 调用 CUDA 函数，传入两个 Hilbert 空间的大小
    return fused_hilbert_pos_embed_cuda(
        coords_s1, 
        // coords_s2, 
        template_s1, 
        // template_s2, 
        batch_size,
        hil_size_z_s1, hil_size_y_s1, hil_size_x_s1,  // hilbert sizes for s1
        // hil_size_z_s2, hil_size_y_s2, hil_size_x_s2,  // hilbert sizes for s2
        sparse_shape_s1,
        // sparse_shape_s2,
        shift
    );
}



// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_window_coors_shift_v2", &get_window_coors_shift_v2, "Get Window Coordinates Shift v2 (CUDA)");
    m.def("flattened_window_mapping", &flattened_window_mapping, "Flattened Window Mapping (CUDA)");
    m.def("get_window_coors_shift_v3", &get_window_coors_shift_v3, "Get Window Coordinates Shift v3 (CUDA)");
    m.def("expand_selected_coords", &expand_selected_coords, "Expand Selected Coordinates (CUDA)");
    m.def("map_points", &map_points, "Map Points (CUDA)");
    m.def("map_points_v2", &map_points_v2, "Map Points v2 (CUDA)");
    m.def("fused_hilbert_pos_embed", &fused_hilbert_pos_embed, "Fused Hilbert Position Embedding (CUDA)");
}