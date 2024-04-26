#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


///////// 3D - > 2D /////////
// RayCasting function s
void raycast_indexer_occ_cuda_forward(
        at::Tensor occ3d,
        at::Tensor occ2d,
        at::Tensor viewMatrixInv,
        at::Tensor intrinsicParams,
        at::Tensor opts,
        at::Tensor mapping3dto2d_num);

void construct_dense_sparse_mapping_cuda(
        torch::Tensor locs,
        torch::Tensor sparse_mapping);

void raycast_indexer_occ_forward(
        at::Tensor occ3d,
        at::Tensor occ2d,
        at::Tensor viewMatrixInv,
        at::Tensor intrinsicParams,
        at::Tensor opts,
        at::Tensor mapping3dto2d_num) {
    CHECK_INPUT(occ3d);
    CHECK_INPUT(occ2d);
    CHECK_INPUT(viewMatrixInv);
    CHECK_INPUT(intrinsicParams);
    CHECK_INPUT(mapping3dto2d_num);

    raycast_indexer_occ_cuda_forward(occ3d, occ2d, viewMatrixInv, intrinsicParams, opts, mapping3dto2d_num);
}

void raycast_indexer_occ_cuda_backward(
        at::Tensor occ3d,
        at::Tensor indexes_image,
        at::Tensor grad_image,
        at::Tensor d_voxels);


///////// 3D - > 2D with Trilinear /////////
void raycast_interpolate_cuda_forward(
        at::Tensor features_3d, // (num_voxels, feature_dim)
        at::Tensor features_2d, // (batch, image_num, image_h, image_w, feature_dim)
        at::Tensor occ3d, // (batch, voxel_z, voxel_y, voxel_x)
        at::Tensor projection_2d, // (batch, image_num, image_h, image_w, 8) for eight neighbouring voxels
        at::Tensor vox_dist_weight, // (batch, image_num, image_h, image_w, 8) for eight neighbouring voxels, storing weights
        at::Tensor viewMatrixInv, // (batch, image_num, 4, 4)
        at::Tensor intrinsicParams, // (batch, [fx,fy,mx,my])
        at::Tensor opts, // depthmin, depthmax, rayIncrement
        at::Tensor mapping3dto2d_num);

void raycast_interpolate_cuda_backward(
        at::Tensor occ3d,
        at::Tensor projection_2d,
        at::Tensor vox_dist_weight,
        at::Tensor grad_image,
        at::Tensor d_voxels);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("raycast_features", &raycast_indexer_occ_forward, "raycast_color 3d occupancy grid to obtain 3D -> 2D index mappings");
  m.def("raycast_features_backward", &raycast_indexer_occ_cuda_backward, "raycast_color 3d occupancy grid to obtan 3D -> 2D index mappings backward function");
  m.def("raycast_interpolate_features", &raycast_interpolate_cuda_forward, "raycast_color 3d occupancy grid to obtain 3D -> 2D index mappings with trilinear interpolation");
  m.def("raycast_interpolate_backward", &raycast_interpolate_cuda_backward, "Backward function after raycasting with trilinear interpolation");
}
