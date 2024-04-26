#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "include/cuda_SimpleMatrixUtil.h"

#include <vector>
#include <cmath>
#include <chrono>

#define T_PER_BLOCK 8
#define IMAGE_T_PER_BLOCK 32

// #define _DEBUG

using namespace torch::indexing;


/////////////////////////////
///////// 2D - > 3D /////////
/////////////////////////////

// Find occupancy Id where ray intersects and copy features to the voxel feature tensor
__device__ void traverseOccGridProjecter(const float *__restrict__ encoded_2d_features, const long *occupancy_3D,
                                         int *mapping2dto3d_num, float *projected_features,
                                         const float3 &worldCamPos, const float3 &worldDir, const float3 &camDir,
                                         const int4 &dTid, const int feature_dim,
                                         const RayCastParams &params,
                                         int dimz, int dimy, int dimx) {

    const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length
    float rayCurrent = depthToRayLength * params.depthMin;    // Convert depth to raylength
    float rayEnd = depthToRayLength * params.depthMax;        // Convert depth to raylength

    int feature_2d_starting_index = -1;
    int occupied_index = 0;

    //Find ray intersection
#pragma unroll 1
    while (rayCurrent < rayEnd) {
        float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
        int3 pos = make_int3(currentPosWorld + make_float3(sign(currentPosWorld)) * 0.5f);
        if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < dimx && pos.y < dimy && pos.z < dimz) {

            occupied_index = occupancy_3D[dTid.w * dimz * dimy * dimx + pos.z * dimy * dimx + pos.y * dimx + pos.x];
            if (occupied_index != 0) {
                feature_2d_starting_index = (dTid.w * params.view_num * params.height * params.width * feature_dim) +
                                            (dTid.z * params.height * params.width * feature_dim) +
                                            (dTid.y * params.width * feature_dim) + (dTid.x * feature_dim);
                atomicAdd(&mapping2dto3d_num[occupied_index], 1);
                break;
            }
        }
        rayCurrent += params.rayIncrement;
    }

    // Iterate over the features at index and copy to output tensor
    if (feature_2d_starting_index != -1 && occupied_index != 0) {
#pragma unroll 1
        for (uint feature_i = 0; feature_i < feature_dim; feature_i++) {
            atomicAdd(&projected_features[occupied_index * feature_dim + feature_i],
                      encoded_2d_features[feature_2d_starting_index + feature_i]);
        }
    }
}

// same as other, with different types and only single feature as pred
__device__ void traverseOccGridPredictionProjecter(const int *__restrict__ predictions_2d,
                                                   const long *occupancy_3D, int *projected_preds,
                                                   const float3 &worldCamPos, const float3 &worldDir,
                                                   const float3 &camDir,
                                                   const int4 &dTid, const int feature_dim,
                                                   const RayCastParams &params,
                                                   int dimz, int dimy, int dimx) {

    const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length
    float rayCurrent = depthToRayLength * params.depthMin;    // Convert depth to raylength
    float rayEnd = depthToRayLength * params.depthMax;        // Convert depth to raylength

    int prediction_2d_starting_index = -1;
    int occupied_index = 0;

    //Find ray intersection
#pragma unroll 1
    while (rayCurrent < rayEnd) {
        float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
        int3 pos = make_int3(currentPosWorld + make_float3(sign(currentPosWorld)) * 0.5f);
        if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < dimx && pos.y < dimy && pos.z < dimz) {

            occupied_index = occupancy_3D[dTid.w * dimz * dimy * dimx + pos.z * dimy * dimx + pos.y * dimx + pos.x];
            if (occupied_index != 0) {
                prediction_2d_starting_index = (dTid.w * params.view_num * params.height * params.width * feature_dim) +
                                               (dTid.z * params.height * params.width * feature_dim) +
                                               (dTid.y * params.width * feature_dim) + (dTid.x * feature_dim);
                break;
            }
        }
        rayCurrent += params.rayIncrement;
    }

    // Iterate over the predictions at index (should be single, but in some cases might be multidimensional
    if (prediction_2d_starting_index != -1 && occupied_index != 0) {
        for (uint pred_i = 0; pred_i < feature_dim; pred_i++) {
            atomicMax(&projected_preds[occupied_index * feature_dim + pred_i],
                      predictions_2d[prediction_2d_starting_index + pred_i]);
        }
    }
}


__global__ void project_features_cuda_forward_kernel(const float *__restrict__ encoded_2d_features,
                                                     const long *__restrict__ occupancy_3D,
                                                     const float *__restrict__ viewMatrixInv,
                                                     const RayCastParams params, const int feature_dim,
                                                     const int dimz, const int dimy, const int dimx,
                                                     int *mapping2dto3d_num,
                                                     float *projected_features) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    if (x < params.width && y < params.height && batch < params.batch_size && view < params.view_num) {

        // Copy the correct view matrix values
        int shift_index = (batch * params.view_num + view) * 16;
        const float4x4 curViewMatrixInv = *(float4x4 *) (viewMatrixInv + shift_index);

        // Calculate ray directions
        float3 camDir = normalize(
                kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch),
                                   params.getFx(batch), params.getFy(batch), x, y, 1.0f));
        float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);
        float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
        float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

        traverseOccGridProjecter(encoded_2d_features, occupancy_3D,
                                 mapping2dto3d_num, projected_features,
                                 worldCamPos, worldDir, camDir,
                                 make_int4(x, y, view, batch), feature_dim,
                                 params,
                                 dimz, dimy, dimx);

    }
}

// same as other, with different types and only single feature as pred
__global__ void project_predictions_cuda_forward_kernel(const int *__restrict__ predictions_2d,
                                                        const long *__restrict__ occupancy_3D,
                                                        const float *__restrict__ viewMatrixInv,
                                                        const RayCastParams params, const int feature_dim,
                                                        const int dimz, const int dimy, const int dimx,
                                                        int *projected_preds) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    if (x < params.width && y < params.height && batch < params.batch_size && view < params.view_num) {


        // Copy the correct view matrix values
        int shift_index = (batch * params.view_num + view) * 16;
        const float4x4 curViewMatrixInv = *(float4x4 *) (viewMatrixInv + shift_index);

        // Calculate ray directions
        float3 camDir = normalize(
                kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch),
                                   params.getFx(batch), params.getFy(batch), x, y, 1.0f));
        float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);
        float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
        float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

        traverseOccGridPredictionProjecter(predictions_2d, occupancy_3D,
                                           projected_preds,
                                           worldCamPos, worldDir, camDir,
                                           make_int4(x, y, view, batch), feature_dim,
                                           params,
                                           dimz, dimy, dimx);

    }
}

void project_features_cuda_forward(at::Tensor encoded_2d_features, // (batch_num, view_num, h_size, w_size, feature_num), dense tensor of features
                                   at::Tensor occupancy_3D, // (batch_num, z_dim, y_dim, x_dim), storing zeros or voxel index as long value, starting from zero with shifted coords
                                   at::Tensor viewMatrixInv,  // (batch_num, view_num, 4, x), view poses, scaled with voxel size and shifted to zero index
                                   at::Tensor intrinsicParams, // (batch_num, 4), [fx, fy, mx, my] scaled with resolution
                                   at::Tensor opts,  // [width, height, depth_min, depth_max, ray_increment]
                                   at::Tensor mapping2dto3d_num, // will be used to normalize projected features in the end
                                   at::Tensor projected_features,  // to store and return feature values on voxels
                                   at::Tensor pred_mode_t) // pred mode enables to use long predictions and pick the max value instead of averaging over all features
{

    // Check device and contiguous memory
    CHECK_INPUT(encoded_2d_features);
    CHECK_INPUT(occupancy_3D);
    CHECK_INPUT(viewMatrixInv);
    CHECK_INPUT(intrinsicParams);
    CHECK_INPUT(mapping2dto3d_num);
    CHECK_INPUT(projected_features);

    // Calculate projection sizes
    int batch_size = encoded_2d_features.size(0);
    int view_size = encoded_2d_features.size(1);
    int voxel_size = projected_features.size(0);
    int image_h = encoded_2d_features.size(2);
    int image_w = encoded_2d_features.size(3);
    int feature_dim = encoded_2d_features.size(4);
    int dim_z = occupancy_3D.size(1);
    int dim_y = occupancy_3D.size(2);
    int dim_x = occupancy_3D.size(3);

    // Init params struct
    auto opts_accessor = opts.accessor<float, 1>();
    RayCastParams params;
    params.width = (int) (opts_accessor[0] + 0.5f);
    params.height = (int) (opts_accessor[1] + 0.5f);
    params.depthMin = opts_accessor[2];
    params.depthMax = opts_accessor[3];
    params.rayIncrement = opts_accessor[4];
    params.intrinsicsParams = intrinsicParams.data<float>();
    params.batch_size = batch_size;
    params.view_num = view_size;

    // get pred mode
    auto pred_mode_accessor = pred_mode_t.accessor<bool, 1>();
    bool pred_mode = pred_mode_accessor[0];

    // Start a thread for every pixel
    const dim3 gridSize((params.width + T_PER_BLOCK - 1) / T_PER_BLOCK, (params.height + T_PER_BLOCK - 1) / T_PER_BLOCK, batch_size);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK, view_size);


    if (! pred_mode){
        project_features_cuda_forward_kernel <<<gridSize, blockSize>>>(encoded_2d_features.data<float>(),
                                                                       occupancy_3D.data<long>(),
                                                                       viewMatrixInv.data<float>(),
                                                                       params, feature_dim,
                                                                       dim_z, dim_y, dim_x,
                                                                       mapping2dto3d_num.data<int>(),
                                                                       projected_features.data<float>());
    }else{ // in prediction mode encoded-2d_features are actually predictions of long type
        project_predictions_cuda_forward_kernel <<<gridSize, blockSize>>>(encoded_2d_features.data<int>(),
                                                                          occupancy_3D.data<long>(),
                                                                          viewMatrixInv.data<float>(),
                                                                          params, feature_dim,
                                                                          dim_z, dim_y, dim_x,
                                                                          projected_features.data<int>());
    }


#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

}



// same as other, with different types and only single feature as pred
__global__ void unproject_depth_images_forward_kernel(const float *__restrict__ depth_image,
                                                      const float *__restrict__ viewMatrixInv,
                                                      const float *__restrict__ intrinsicParams,
                                                      float *batched_point_cloud,
                                                      const int height, const int width, const int view_num) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int view = blockIdx.z;

    if (x < width && y < height && view < view_num) {

        const long point_cloud_index = view * height * width + y * width + x;

        // Get the depth value
        const float depth_value = depth_image[point_cloud_index];
        if (depth_value <= 0) { return; } // don't project if invalid point

        // Copy the correct view matrix values
        const int view_shift = view * 16;
        const int intrinsict_shift = view * 4;
        const float4x4 curViewMatrixInv = *(float4x4 *) (viewMatrixInv + view_shift);

        const float fx = intrinsicParams[intrinsict_shift];
        const float fy = intrinsicParams[intrinsict_shift + 1];
        const float mx = intrinsicParams[intrinsict_shift + 2];
        const float my = intrinsicParams[intrinsict_shift + 3];

        const float dx = ((float)x - mx) * depth_value / fx;
        const float dy = ((float)y - my) * depth_value / fy;
        const float3 point_pos = make_float3(dx, dy, depth_value);

        // Finally get the point position of depth value and transform to world space
        float3 world_pos = curViewMatrixInv * point_pos;

        // update batch and xyz coordinates in the output
        batched_point_cloud[point_cloud_index * 5] = (float)view;
        batched_point_cloud[point_cloud_index * 5 + 1] = (float)point_cloud_index;
        batched_point_cloud[point_cloud_index * 5 + 2] = world_pos.x;
        batched_point_cloud[point_cloud_index * 5 + 3] = world_pos.y;
        batched_point_cloud[point_cloud_index * 5 + 4] = world_pos.z;
    }
}

void unproject_depth_images(
        at::Tensor depth_images, // (view_num, h_size, w_size), dense tensor of depth values for all images
        at::Tensor viewMatrixInv, //  (view_num, 4, 4) world to camera pose
        at::Tensor intrinsicParams, // (view_num, 4), [fx, fy, mx, my] scaled with resolution
        at::Tensor batched_point_cloud) // (point_num, 4) [view_id, x, y, z] batched coordinates
{

    // Check device and contiguous memory
    CHECK_INPUT(depth_images);
    CHECK_INPUT(viewMatrixInv);
    CHECK_INPUT(intrinsicParams);
    CHECK_INPUT(batched_point_cloud);

    // Calculate projection sizes
    int view_num = depth_images.size(0);
    int image_h = depth_images.size(1);
    int image_w = depth_images.size(2);


    // Start a thread for every pixel
    const dim3 gridSize((image_w + T_PER_BLOCK - 1) / T_PER_BLOCK, (image_h + T_PER_BLOCK - 1) / T_PER_BLOCK, view_num);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);


    unproject_depth_images_forward_kernel <<<gridSize, blockSize>>>(depth_images.data<float>(),
                                                                    viewMatrixInv.data<float>(),
                                                                    intrinsicParams.data<float>(),
                                                                    batched_point_cloud.data<float>(),
                                                                    image_h, image_w, view_num);


    #ifdef _DEBUG
            cutilSafeCall(cudaDeviceSynchronize());
            cutilCheckMsg(__FUNCTION__);
    #endif

    }

