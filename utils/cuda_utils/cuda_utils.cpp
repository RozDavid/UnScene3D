#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// Interpolation functions from dense Tensor to query points
// Similar to the MinkowskiEngine Interpolation, but doesn't take into account unoccupied vertices
void trilinear_interpolate_forward(at::Tensor occupancy_3d,
                                   at::Tensor features_3d,
                                   at::Tensor query_coords,
                                   at::Tensor query_features,
                                   at::Tensor interpolation_indices,
                                   at::Tensor interpolation_weights,
                                   at::Tensor accum_voxel_weights);

void trilinear_interpolate_backward(at::Tensor interpolation_indices,
                                    at::Tensor interpolation_weights,
                                    at::Tensor accum_voxel_weights,
                                    at::Tensor query_grads,
                                    at::Tensor voxel_grads);

// Project Sparse with zero min values into XY, XY, and YZ planes for noise robust losses
void project_sparse_voxels_to_planes(at::Tensor s_coords,
                                     at::Tensor s_predictions,
                                     at::Tensor s_targets,
                                     at::Tensor xy_pred_projections,
                                     at::Tensor xz_pred_projections,
                                     at::Tensor yz_pred_projections,
                                     at::Tensor xy_target_projections,
                                     at::Tensor xz_target_projections,
                                     at::Tensor yz_target_projections,
                                     at::Tensor xy_projection_nums,
                                     at::Tensor xz_projection_nums,
                                     at::Tensor yz_projection_nums);

void project_sparse_voxels_to_planes_backward(at::Tensor s_coords,
                                              at::Tensor s_grads,
                                              at::Tensor xy_grads,
                                              at::Tensor xz_grads,
                                              at::Tensor yz_grads,
                                              at::Tensor xy_nums,
                                              at::Tensor xz_nums,
                                              at::Tensor yz_nums);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_interpolate", &trilinear_interpolate_forward, "Interpolate features for query coordinates from occupied source voxels");
    m.def("trilinear_interpolate_backward", &trilinear_interpolate_backward, "Backward function for Trilinear interpolation");
    m.def("project_sparse_voxels_to_planes", &project_sparse_voxels_to_planes, "Project Sparse with zero min values into XY, XY, and YZ planes for noise robust losses");
    m.def("project_sparse_voxels_to_planes_backward", &project_sparse_voxels_to_planes_backward, "Backward function for projection based supervision");
}
