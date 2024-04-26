#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

///////// 2D - > 3D /////////
// CUDA forward declaration
void project_features_cuda_forward(at::Tensor encoded_2d_features,
                                   at::Tensor occupancy_3D,
                                   at::Tensor viewMatrixInv,
                                   at::Tensor intrinsicParams,
                                   at::Tensor opts,
                                   at::Tensor mapping2dto3d_num,
                                   at::Tensor projected_features,
                                   at::Tensor pred_mode_t);

// take a batch if images and unporject valid pixels to 3d
void unproject_depth_images(at::Tensor depth_images,
                           at::Tensor viewMatrixInv,
                           at::Tensor intrinsicParams,
                           at::Tensor batched_point_cloud);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_features_cuda", &project_features_cuda_forward, "Projecting from 2D to 3D (With CUDA kernels)");
    m.def("unproject_depth_images", &unproject_depth_images, "Projecting from 2D to 3D a batch of depth images with camera poses and parameters");
}
