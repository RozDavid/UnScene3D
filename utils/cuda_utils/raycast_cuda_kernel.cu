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
///////// 3D - > 2D /////////
/////////////////////////////

// Indexer versions of raycasing
__device__ void traverseOccGridIndexer(const long *occ3d, long *occ2d, int *mapping3dto2d_num,  const float3 &worldCamPos, const float3 &worldDir,
                            const float3 &camDir, const int4 &dTid, const RayCastParams &params,
                            int dimz, int dimy, int dimx) {
    const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length
    float rayCurrent = depthToRayLength * params.depthMin;    // Convert depth to raylength
    float rayEnd = depthToRayLength * params.depthMax;        // Convert depth to raylength
#pragma unroll 1
    while (rayCurrent < rayEnd) {
        float3 currentPosWorld = worldCamPos + rayCurrent * worldDir;
        int3 pos = make_int3(currentPosWorld + make_float3(sign(currentPosWorld)) * 0.5f);
        if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < dimx && pos.y < dimy && pos.z < dimz) {

            int occupied_index = occ3d[dTid.w * dimz * dimy * dimx + pos.z * dimy * dimx + pos.y * dimx + pos.x];
            if (occupied_index != 0) {
                occ2d[dTid.w * params.view_num * params.width * params.height + dTid.z * params.width * params.height + dTid.y * params.width + dTid.x] = occupied_index;
                atomicAdd(&mapping3dto2d_num[occupied_index], 1);
                return;
            }
        }
        rayCurrent += params.rayIncrement;
    }
}

__global__ void raycast_indexer_occ_cuda_kernel(
        const long *__restrict__ occ3d,
        long *__restrict__ occ2d,
        int *__restrict__ mapping3dto2d_num,
        const float *__restrict__ viewMatrixInv,
        const RayCastParams params,
        int dimz,
        int dimy,
        int dimx) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    if (x < params.width && y < params.height && batch < params.batch_size && view < params.view_num) {

        // init with ignore label
        occ2d[batch * params.view_num * params.width * params.height + view * params.width + params.height + y * params.width + x] = -1;

        //Copy the correct view matrix values
        int shift_index = (batch * params.view_num + view) * 16;
        const float4x4 curViewMatrixInv = *(float4x4 *) (viewMatrixInv + shift_index);

        float3 camDir = normalize(
                kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch),
                                   params.getFx(batch), params.getFy(batch), x, y, 1.0f));
        float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);

        float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
        float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

        traverseOccGridIndexer(occ3d, occ2d, mapping3dto2d_num, worldCamPos, worldDir, camDir, make_int4(x, y, view, batch), params, dimz, dimy,
                               dimx);
    }
}


void raycast_indexer_occ_cuda_forward(
        at::Tensor occ3d, // (batch, voxel_z, voxel_y, voxel_x)
        at::Tensor occ2d, // (batch, image_num, image_h, image_w)
        at::Tensor viewMatrixInv, // (batch, image_num, 4, 4)
        at::Tensor intrinsicParams, // (batch, [fx,fy,mx,my])
        at::Tensor opts, // depthmin, depthmax, rayIncrement
        at::Tensor mapping3dto2d_num) { // number of 3D -> 2D associations (all_voxel_num,)

    // Get dimensions
    const int batch_size = occ2d.size(0);
    const int view_num = occ2d.size(1);
    const int dimz = occ3d.size(1);
    const int dimy = occ3d.size(2);
    const int dimx = occ3d.size(3);

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
    params.view_num = view_num;

    const dim3 gridSize((params.width + T_PER_BLOCK - 1) / T_PER_BLOCK, (params.height + T_PER_BLOCK - 1) / T_PER_BLOCK, batch_size);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK, view_num);

    raycast_indexer_occ_cuda_kernel<<<gridSize, blockSize>>>(
            occ3d.data<long>(),
            occ2d.data<long>(),
            mapping3dto2d_num.data<int>(),
            viewMatrixInv.data<float>(),
            params,
            dimz,
            dimy,
            dimx);

    #ifdef _DEBUG
        cutilSafeCall(cudaDeviceSynchronize());
        cutilCheckMsg(__FUNCTION__);
    #endif
}


__global__ void raycast_indexer_occ_cuda_backward_kernel(
        const long *__restrict__ occ3d,
        const long *__restrict__ indexes_image,  // containing the voxel index for every assigned pixel
        const float *__restrict__ grad_image,
        float *__restrict__ d_voxels,
        const int batch_size, const int view_size, const int image_h, const int image_w,
        const int num_labels, const int num_voxels){

    const unsigned int w_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int h_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    // Skip if indexed out to illegal
    if (h_index >= image_h || w_index >= image_w || batch >= batch_size || view >= view_size) { return; }

    const unsigned long index_image_idx = (batch * view_size * image_h * image_w) + (view * image_h * image_w) + (h_index * image_w) + (w_index);
    const unsigned long voxel_index = indexes_image[index_image_idx];

    // return for non-projected voxel
    if (voxel_index >= num_voxels){
        return;
    }

    // Update the voxel grad value with all associated pixel grads and labels - > will be normalized after the kernel
    #pragma unroll 1
    for (int category_index = 0; category_index < num_labels; category_index++) {
        atomicAdd(&d_voxels[voxel_index * num_labels + category_index], grad_image[index_image_idx * num_labels + category_index]);
    }
}

void raycast_indexer_occ_cuda_backward(
        at::Tensor occ3d,
        at::Tensor indexes_image,
        at::Tensor grad_image,
        at::Tensor d_voxels){

    const int batch_size = indexes_image.size(0);
    const int view_size = indexes_image.size(1);
    const int image_h = indexes_image.size(2);
    const int image_w = indexes_image.size(3);
    const int num_voxels = d_voxels.size(0);
    const int num_labels = d_voxels.size(1);

    const dim3 gridSize((image_w + T_PER_BLOCK - 1) / T_PER_BLOCK, (image_h + T_PER_BLOCK - 1) / T_PER_BLOCK, batch_size);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK, view_size);

    // Run kernel for updating the values
    raycast_indexer_occ_cuda_backward_kernel<<<gridSize, blockSize>>>(
            occ3d.data<long>(),
            indexes_image.data<long>(),
            grad_image.data<float>(),
            d_voxels.data<float>(),
            batch_size, view_size, image_h, image_w,
            num_labels, num_voxels);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


///////////////////////////////
/// Trilinear interpolation ///
///////////////////////////////
// Indexer versions of with trilinear interpolation
__device__ void traverseOccgridIndexInterpolate(const long *occ3d, long *projection_2d, float *vox_dist_weight, float *mapping3dto2d_num,  const float3 &worldCamPos, const float3 &worldDir,
                                                const float3 &camDir, const int4 &dTid, const RayCastParams &params,
                                                int dimz, int dimy, int dimx) {

    const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length
    float rayCurrent = depthToRayLength * params.depthMin;    // Convert depth to raylength
    float rayEnd = depthToRayLength * params.depthMax;        // Convert depth to raylength

#pragma unroll 1
    while (rayCurrent < rayEnd) {
        float3 currentPosWorld = worldCamPos + rayCurrent * worldDir; // exact position the ray sample is located
        int3 voxel_pos =  make_int3( currentPosWorld ); // check if the rounded-down position is occupied

        if (currentPosWorld.x >= 0 && currentPosWorld.y >= 0 && currentPosWorld.z >= 0 && voxel_pos.x < dimx && voxel_pos.y < dimy && voxel_pos.z < dimz) {

            int occupied_index = occ3d[dTid.w * dimz * dimy * dimx + voxel_pos.z * dimy * dimx + voxel_pos.y * dimx + voxel_pos.x]; // check occupation for rounded
            if (occupied_index != 0) { // unoccupied voxels are indexed with 0

                //Check direction in closes neighbours
                //int3 signed_offset = sign(worldDir);  // doesnt work
                //int3 signed_offset = sign(currentPosWorld - make_float3(voxel_pos.x, voxel_pos.y, voxel_pos.z));  // offset is the ray direction in world space // doesnt work
                const int3 signed_offset = make_int3(1, 1, 1);

                const float voxel_max_dist = sqrt(3);
                float eps = 10e-5;

                float weight_sum = eps; // to normalize the voxel distances in the end to be 1
                const long projection_index_shift = (dTid.w * params.view_num * params.width * params.height + dTid.z * params.width * params.height + dTid.y * params.width + dTid.x) * 8;  // we are storing 8 values for every pixel for teh 8 neighbouring voxels

                // calculate distance (weight) and voxel index for all neighbouring voxel
                // the weights are calculated by the distance to voxel
                // mappings are stored in the projection_2d array
                // weights are summed up based on distance to make up to 1 for every pixel
                int offset_shift = 0;
                float dist = length(currentPosWorld - make_float3(voxel_pos.x, voxel_pos.y, voxel_pos.z));
                float weight0 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                weight_sum += weight0;
                atomicAdd(&mapping3dto2d_num[occupied_index], weight0);
                projection_2d[projection_index_shift + offset_shift] = occupied_index;  // zeros is without offset

                // N1
                float weight1 = 0.;
                offset_shift++;
                int3 n_voxel = voxel_pos + make_int3(signed_offset.x, 0, 0);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight1 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight1;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight1);
                    }
                }

                // N2
                float weight2 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(0, signed_offset.y, 0);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight2 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight2;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight2);
                    }
                }

                // N3
                float weight3 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(0, 0, signed_offset.z);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight3 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight3;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight3);
                    }
                }

                // N4
                float weight4 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(signed_offset.x, signed_offset.y, 0);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight4 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight4;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight4);
                    }
                }

                // N5
                float weight5 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(0, signed_offset.y, signed_offset.z);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight5 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight5;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight5);
                    }
                }

                // N6
                float weight6 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(signed_offset.x, 0, signed_offset.z);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0){
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight6 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight6;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight6);
                    }
                }

                // N7
                float weight7 = 0.;
                offset_shift++;
                n_voxel = voxel_pos + make_int3(signed_offset.x, signed_offset.y, signed_offset.z);
                if (n_voxel.x < dimx && n_voxel.y < dimy && n_voxel.z < dimz) {
                    occupied_index = occ3d[dTid.w * dimz * dimy * dimx + n_voxel.z * dimy * dimx + n_voxel.y * dimx + n_voxel.x];
                    if (occupied_index != 0) {
                        float dist = length(currentPosWorld - make_float3(n_voxel.x, n_voxel.y, n_voxel.z));
                        weight7 = fmaxf((voxel_max_dist - dist), 0) / voxel_max_dist;
                        projection_2d[projection_index_shift + offset_shift] = occupied_index;  // shift with current neighbour id
                        weight_sum += weight7;
                        atomicAdd(&mapping3dto2d_num[occupied_index], weight7);
                    }
                }

                // Add normalize weights finally to both 2D and 3D containers
                if (weight_sum > eps){
                    vox_dist_weight[projection_index_shift] = weight0 / weight_sum;
                    vox_dist_weight[projection_index_shift + 1] = weight1 / weight_sum;
                    vox_dist_weight[projection_index_shift + 2] = weight2 / weight_sum;
                    vox_dist_weight[projection_index_shift + 3] = weight3 / weight_sum;
                    vox_dist_weight[projection_index_shift + 4] = weight4 / weight_sum;
                    vox_dist_weight[projection_index_shift + 5] = weight5 / weight_sum;
                    vox_dist_weight[projection_index_shift + 6] = weight6 / weight_sum;
                    vox_dist_weight[projection_index_shift + 7] = weight7 / weight_sum;
                }
                return;
            }
        }
        rayCurrent += params.rayIncrement;
    }
}


__global__ void weighted_copy_features_3d_to_2d(const float *features_3d, float *features_2d, const long *projection_2d, const float *vox_dist_weight,
                                                const int batch_size, const int view_size, const int img_height, const int img_width, const int feature_dim, const int voxel_num){

    // get thread ids
    const unsigned int width = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int height = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int view = threadIdx.z;

    // return if invalid
    if (height >= img_height || width >= img_width || batch >= batch_size || view >= view_size) { return; }

    // get the array shift indexes for both the indexer and the feature tensor
    long projection_index_shift = (batch * view_size * img_height * img_width + view * img_height * img_width + height * img_width + width) * 8;  // we are storing 8 values for every pixel for teh 8 neighbouring voxels
    long projection_feature_shift = (batch * view_size * img_height * img_width + view * img_height * img_width + height * img_width + width) * feature_dim;  // for every pixel we project the 3d feature

    // for all neighbours
    for (int n = 0; n < 8; n++){
        long voxel_index = projection_2d[projection_index_shift + n];
        float voxel_weight = vox_dist_weight[projection_index_shift + n];

        if (voxel_index < 0 || voxel_index > voxel_num){ continue; }

        // for all features add the 3d value to 2d weighted with the voxel distance
        #pragma unroll 1
        for (int f = 0; f < feature_dim; f++){
            atomicAdd(&features_2d[projection_feature_shift + f], (features_3d[voxel_index * feature_dim + f] * voxel_weight));
        }
    }

}

__global__ void raycast_interpolate_cuda_kernel(
        const long *__restrict__ occ3d,
        long *__restrict__ projection_2d,
        float *__restrict__ vox_dist_weight,
        float *__restrict__ mapping3dto2d_num,
        const float *__restrict__ viewMatrixInv,
        const RayCastParams params,
        int dimz,
        int dimy,
        int dimx) {

    // get thread ids
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    // projection_2d_tensor should be initialized before passing to cuda kernels
    if (x < params.width && y < params.height && batch < params.batch_size && view < params.view_num) {

        //Copy the correct view matrix values
        int shift_index = (batch * params.view_num + view) * 16;
        const float4x4 curViewMatrixInv = *(float4x4 *) (viewMatrixInv + shift_index);

        float3 camDir = normalize(
                kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch),
                                   params.getFx(batch), params.getFy(batch), x, y, 1.0f));
        float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);

        float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
        float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

        traverseOccgridIndexInterpolate(occ3d, projection_2d, vox_dist_weight, mapping3dto2d_num, worldCamPos, worldDir, camDir, make_int4(x, y, view, batch), params, dimz, dimy, dimx);
    }
}


void raycast_interpolate_cuda_forward(
        at::Tensor features_3d, // (num_voxels, feature_dim)
        at::Tensor features_2d, // (batch, image_num, image_h, image_w, feature_dim)
        at::Tensor occ3d, // (batch, voxel_z, voxel_y, voxel_x)
        at::Tensor projection_2d, // (batch, image_num, image_h, image_w, 8) for eight neighbouring voxels
        at::Tensor vox_dist_weight, // (batch, image_num, image_h, image_w, 8) for eight neighbouring voxels, storing weights
        at::Tensor viewMatrixInv, // (batch, image_num, 4, 4)
        at::Tensor intrinsicParams, // (batch, [fx,fy,mx,my])
        at::Tensor opts, // depthmin, depthmax, rayIncrement
        at::Tensor mapping3dto2d_num) { // (num_voxels,)  number of 3D -> 2D associations (float tensor for)

    // Get dimensions
    const int batch_size = projection_2d.size(0);
    const int view_num = projection_2d.size(1);
    const int dimz = occ3d.size(1);
    const int dimy = occ3d.size(2);
    const int dimx = occ3d.size(3);
    const int feature_dim = features_3d.size(1);
    const int voxel_num = features_3d.size(0);

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
    params.view_num = view_num;

    const dim3 gridSize((params.width + T_PER_BLOCK - 1) / T_PER_BLOCK, (params.height + T_PER_BLOCK - 1) / T_PER_BLOCK, batch_size);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK, view_num);

    raycast_interpolate_cuda_kernel<<<gridSize, blockSize>>>(
            occ3d.data<long>(),
            projection_2d.data<long>(),
            vox_dist_weight.data<float>(),
            mapping3dto2d_num.data<float>(),
            viewMatrixInv.data<float>(),
            params,
            dimz,
            dimy,
            dimx);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

    // summarize features with weights
    weighted_copy_features_3d_to_2d<<<gridSize, blockSize>>>(
            features_3d.data<float>(),
            features_2d.data<float>(),
            projection_2d.data<long>(),
            vox_dist_weight.data<float>(),
            params.batch_size, params.view_num, params.height, params.width,
            feature_dim, voxel_num);

#ifdef _DEBUG
        cutilSafeCall(cudaDeviceSynchronize());
        cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void raycast_interpolate_cuda_backward_kernel(
        const long *__restrict__ occ3d,
        const long *__restrict__ projection_2d,  // storing associated voxel ids for all neighbours
        const float *__restrict__ vox_dist_weight, // storing voxel weights for all pixels with neighbours
        const float *__restrict__ grad_image,
        float *__restrict__ d_voxels,
        const int batch_size, const int view_size, const int image_h, const int image_w,
        const int num_features, const int num_voxels){

    const unsigned int w_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int h_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.z;
    const int view = threadIdx.z;

    // Skip if indexed out to illegal
    if (h_index >= image_h || w_index >= image_w || batch >= batch_size || view >= view_size) { return; }

    const unsigned long grad_image_shift = (batch * view_size * image_h * image_w) + (view * image_h * image_w) + (h_index * image_w) + (w_index);
    const unsigned long projection_2d_img_shift = grad_image_shift * 8;

    // repeat for all neighbours to add gradients weighted with voxel distance at pixel
    unsigned long voxel_index = 0;
    float voxel_weight = 0.0;
    for (int n = 0; n < 8; n++) {

        // get voxel index and weight of actual and neighbouring voxel
        voxel_index = projection_2d[projection_2d_img_shift + n];
        voxel_weight = vox_dist_weight[projection_2d_img_shift + n];

        // return for non-projected voxel
        if (voxel_index >= num_voxels){
            continue;
        }

        // Update the voxel grad value with all associated pixel grads and labels - > will be normalized after the kernel
        #pragma unroll 1
        for (int category_index = 0; category_index < num_features; category_index++) {
            atomicAdd(&d_voxels[voxel_index * num_features + category_index],
                      grad_image[grad_image_shift * num_features + category_index] * voxel_weight);
        }
    }
}


void raycast_interpolate_cuda_backward(
        at::Tensor occ3d,
        at::Tensor projection_2d,
        at::Tensor vox_dist_weight,
        at::Tensor grad_image,
        at::Tensor d_voxels){

    const int batch_size = projection_2d.size(0);
    const int view_size = projection_2d.size(1);
    const int image_h = projection_2d.size(2);
    const int image_w = projection_2d.size(3);
    const int num_voxels = d_voxels.size(0);
    const int num_features = d_voxels.size(1);

    const dim3 gridSize((image_w + T_PER_BLOCK - 1) / T_PER_BLOCK, (image_h + T_PER_BLOCK - 1) / T_PER_BLOCK, batch_size);
    const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK, view_size);

    // Run kernel for updating the values
    raycast_interpolate_cuda_backward_kernel<<<gridSize, blockSize>>>(
            occ3d.data<long>(),
            projection_2d.data<long>(),
            vox_dist_weight.data<float>(),
            grad_image.data<float>(),
            d_voxels.data<float>(),
            batch_size, view_size, image_h, image_w,
            num_features, num_voxels);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}
