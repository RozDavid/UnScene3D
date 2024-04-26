#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "include/cuda_SimpleMatrixUtil.h"

#include <vector>
#include <cmath>
#include <chrono>

#define _DEBUG
#define T_PER_BLOCK 512

using namespace torch::indexing;

/// Helper functions ///
void __device__ coordinate_to_tensor_index(const unsigned int batch_id, const int3 coord,
                                           const int dimx, const int dimy, const int dimz, long *occupancy_id) {
    *occupancy_id = (batch_id * dimx * dimy * dimz) + (coord.x * dimy * dimz) + (coord.y * dimz) + coord.z;
}

/// Forward functions for trilinear interpolation///
void __global__ trilinear_interpolate_forward_kernel(
        const long *__restrict__ occupancy_3d,
        const float *__restrict__ features_3d,
        const float *__restrict__ query_coords,
        float *query_features,
        long *interpolation_indices,
        float *interpolation_weights,
        float *accum_voxel_weights,
        const int batch_size, const int query_num, const int souce_voxel_num,
        const int feature_dim, const int query_dim,
        const int dimx, const int dimy, const int dimz) {

    // This should be the voxel diagonal, and other hyperparameters for regularization
    const float voxel_max_dist = sqrt(3);
    const float eps = 10e-5;
    float weight_sum = eps;
    float dist, weight;

    //Get voxel index and position
    const unsigned int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch = (unsigned int) query_coords[query_id * 4];  // coords are all of the points batched, where the first element is the batch id in float
    const unsigned int neighbor_ind_shift = query_id * 8;
    const unsigned int query_feat_ind_shift = query_id * feature_dim;

    // return for invalid voxels and non 3D coords
    if (query_id >= query_num || query_dim != 4) { return; }

    // get query point location - we only work with 3D coords here
    const float3 query_coord = make_float3(query_coords[query_id * query_dim + 1],
                                           query_coords[query_id * query_dim + 2],
                                           query_coords[query_id * query_dim + 3]);
    //// N0 ///
    // Add weights with relative distance between sample point and voxel vertices
    int3 voxel_coord = make_int3(query_coord.x, query_coord.y, query_coord.z);

    // check if voxel is within limits - this is happening for all neighbours
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    long voxel_index;
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    long occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift] = weight;
        interpolation_indices[neighbor_ind_shift] = occ_value;
    }


    //// N1 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 1] = weight;
        interpolation_indices[neighbor_ind_shift + 1] = occ_value;
    }

    //// N2 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y + 1, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 2] = weight;
        interpolation_indices[neighbor_ind_shift + 2] = occ_value;
    }

    //// N3 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 3] = weight;
        interpolation_indices[neighbor_ind_shift + 3] = occ_value;
    }

    //// N4 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y + 1, query_coord.z);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 4] = weight;
        interpolation_indices[neighbor_ind_shift + 4] = occ_value;
    }

    //// N5 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x, query_coord.y + 1, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 5] = weight;
        interpolation_indices[neighbor_ind_shift + 5] = occ_value;
    }

    //// N6 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 6] = weight;
        interpolation_indices[neighbor_ind_shift + 6] = occ_value;
    }

    //// N7 ///
    // Add weights with relative distance between sample point and voxel vertice
    voxel_coord = make_int3(query_coord.x + 1, query_coord.y + 1, query_coord.z + 1);

    // check if voxel is within limits
    if (query_coord.x < 0 || query_coord.y < 0 || query_coord.z < 0 || voxel_coord.x >= dimx || voxel_coord.y >= dimy ||
        voxel_coord.z >= dimz) { return; }

    // check if voxel is occupied
    coordinate_to_tensor_index(batch, voxel_coord, dimx, dimy, dimz, &voxel_index);
    occ_value = occupancy_3d[voxel_index];
    if (occ_value >= 0) {
        dist = length(query_coord - make_float3(voxel_coord.x, voxel_coord.y, voxel_coord.z));
        weight = (voxel_max_dist - dist) / voxel_max_dist;
        weight_sum += weight;
        interpolation_weights[neighbor_ind_shift + 7] = weight;
        interpolation_indices[neighbor_ind_shift + 7] = occ_value;
    }

    if (weight_sum == eps) { return; } // no neighbour was used

    // Normalize weights with sum
    interpolation_weights[neighbor_ind_shift + 0] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 1] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 2] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 3] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 4] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 5] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 6] /= weight_sum;
    interpolation_weights[neighbor_ind_shift + 7] /= weight_sum;

    // iterate over all neighbours and copy voxel features to query features
    for (int n = 0; n < 8; n++) {

        const long voxel_id = interpolation_indices[neighbor_ind_shift + n];
        const float voxel_weight = interpolation_weights[neighbor_ind_shift + n];

        // check if voxel index is meaningful - dont update accumulators for unobserved regions
        if (voxel_id < 0 || voxel_weight == 0) { continue; }

        // update the voxel weight accumulator
        atomicAdd(&accum_voxel_weights[voxel_id], voxel_weight);

        // get feature index of the voxel space
        const long feature_shift_index = voxel_id * feature_dim;

        // iterate over all features
#pragma unroll 1
        for (int f = 0; f < feature_dim; f++) {
            atomicAdd(&query_features[query_feat_ind_shift + f], features_3d[feature_shift_index + f] * voxel_weight);
        }

    }
}

void trilinear_interpolate_forward(at::Tensor occupancy_3d, // (batch_size, dim_x, dim_y, dim_z)
                                   at::Tensor features_3d,  // SparseTensor.F -- indexes are stored in occupancy_3d
                                   at::Tensor query_coords, // coordinate locations of the query field
                                   at::Tensor query_features, // to be used to copy query features to
                                   at::Tensor interpolation_indices, // to be used to store interpolation indices for backward
                                   at::Tensor interpolation_weights, // to be used to store interpolation voxel weights
                                   at::Tensor accum_voxel_weights)  //storing the weighted number of voxel-point associations to use for voxel grad normalization in backward pass

{
    // Check inputs
    CHECK_INPUT(occupancy_3d);
    CHECK_INPUT(features_3d);
    CHECK_INPUT(query_coords);
    CHECK_INPUT(query_features);
    CHECK_INPUT(interpolation_indices);
    CHECK_INPUT(interpolation_weights);
    CHECK_INPUT(accum_voxel_weights);

    // Get dimensions
    const int batch_size = occupancy_3d.size(0);
    const int query_num = query_coords.size(0);
    const int feature_dim = features_3d.size(1);
    const int souce_voxel_num = features_3d.size(0);
    const int query_dim = query_coords.size(1);
    const int dimx = occupancy_3d.size(1);
    const int dimy = occupancy_3d.size(2);
    const int dimz = occupancy_3d.size(3);

    const dim3 gridSize((query_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    trilinear_interpolate_forward_kernel<<<gridSize, blockSize>>>(
            occupancy_3d.data<long>(),
            features_3d.data<float>(),
            query_coords.data<float>(),
            query_features.data<float>(),
            interpolation_indices.data<long>(),
            interpolation_weights.data<float>(),
            accum_voxel_weights.data<float>(),
            batch_size, query_num, souce_voxel_num,
            feature_dim, query_dim,
            dimx, dimy, dimz);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


/// Backward functions ///
void __global__ trilinear_interpolate_backward_kernel(const long *__restrict__ interpolation_indices,
                                                      const float *__restrict__ interpolation_weights,
                                                      const float *__restrict__ accum_voxel_weights,
                                                      const float *__restrict__ query_grad,
                                                      float *voxel_grad,
                                                      const int query_num, const int grad_dim, const int voxel_num){

    // Get voxel index and position
    const unsigned int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int weight_ind_shift = query_id * 8;
    const unsigned int query_grad_shift = query_id * grad_dim;

    // return for invalid query points
    if (query_id >= query_num || query_id < 0) { return; }

    // For all neighbours do the grad summary
    int voxel_index, n_grad_shift;
    float n_weight, voxel_accum_weight;
    for (int n = 0; n < 8; n++){

        voxel_index = interpolation_indices[weight_ind_shift + n];
        n_weight = interpolation_weights[weight_ind_shift + n];
        n_grad_shift = voxel_index * grad_dim;

        // if neighbour was not initialized during interpolation skip, or index is invalid return
        if (voxel_index < 0 || voxel_index >= voxel_num) { continue; }

        // Get the accumulated weight only of it is valid voxel index
        voxel_accum_weight = accum_voxel_weights[voxel_index];
        if (voxel_accum_weight <= 0.) { continue; }

        // for all grad dimensions add the value
#pragma unroll 1
        for (int f = 0; f < grad_dim; f++){
            atomicAdd(&voxel_grad[n_grad_shift + f], query_grad[query_grad_shift + f] * n_weight / voxel_accum_weight);
        }
    }
}

void trilinear_interpolate_backward(at::Tensor interpolation_indices,
                                    at::Tensor interpolation_weights,
                                    at::Tensor accum_voxel_weights,
                                    at::Tensor query_grads,
                                    at::Tensor voxel_grads){

    // Check inputs
    CHECK_INPUT(interpolation_indices);
    CHECK_INPUT(interpolation_weights);
    CHECK_INPUT(accum_voxel_weights);
    CHECK_INPUT(query_grads);
    CHECK_INPUT(voxel_grads);

    // Get dimensions
    const unsigned int query_num = query_grads.size(0);
    const unsigned int grad_dim = query_grads.size(1);
    const unsigned int voxel_num = voxel_grads.size(0);

    const dim3 gridSize((query_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    trilinear_interpolate_backward_kernel<<<gridSize, blockSize>>>(
            interpolation_indices.data<long>(),
            interpolation_weights.data<float>(),
            accum_voxel_weights.data<float>(),
            query_grads.data<float>(),
            voxel_grads.data<float>(),
            query_num, grad_dim, voxel_num);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

}


void __global__ project_sparse_voxels_to_planes_kernel(const int* s_coords,
                                                       const float* s_predictions,
                                                       const float* s_targets,
                                                       float* xy_pred_projections,
                                                       float* xz_pred_projections,
                                                       float* yz_pred_projections,
                                                       float* xy_target_projections,
                                                       float* xz_target_projections,
                                                       float* yz_target_projections,
                                                       int* xy_projection_nums,
                                                       int* xz_projection_nums,
                                                       int* yz_projection_nums,
                                                       const int voxel_num, const int inst_num,
                                                       const int x_dim, const int y_dim, const int z_dim) {

    //Get voxel index and position
    const unsigned int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;

    // return for invalid voxels
    if (voxel_id >= voxel_num) { return; }

    const int coord_x = s_coords[voxel_id * 4 + 1];
    const int coord_y = s_coords[voxel_id * 4 + 2];
    const int coord_z = s_coords[voxel_id * 4 + 3];

    // This should be an error - views not correctly initialized
    if (coord_x >= x_dim || coord_y >= y_dim || coord_z >= z_dim || coord_x  < 0 || coord_y  < 0 || coord_z < 0) { return; }

    // We don't have instance dimension in nums as that is the same for all of those
    const unsigned int vox_num_shift_xy = coord_x * y_dim + coord_y;
    const unsigned int vox_num_shift_xz = coord_x * z_dim + coord_z;
    const unsigned int vox_num_shift_yz = coord_y * z_dim + coord_z;

    // Update the projection nums for all 2D pixels
    atomicAdd(&xy_projection_nums[vox_num_shift_xy], 1);
    atomicAdd(&xz_projection_nums[vox_num_shift_xz], 1);
    atomicAdd(&yz_projection_nums[vox_num_shift_yz], 1);
    __syncthreads();

    // For valid voxels iterate over all instance predictions/targets
    for (int pred_id = 0; pred_id < inst_num; pred_id++) {

        // Bet location shifts with instance id number
        const unsigned int pred_target_id = voxel_id * inst_num + pred_id;
        const unsigned int instance_shift_xy = vox_num_shift_xy * inst_num + pred_id;
        const unsigned int instance_shift_xz = vox_num_shift_xz * inst_num + pred_id;
        const unsigned int instance_shift_yz = vox_num_shift_yz * inst_num + pred_id;

        // The actual values to project
        const float pred_value = s_predictions[pred_target_id];
        const float target_value = s_targets[pred_target_id];

        // Add the predictions to the corresponding views
        atomicAdd(&xy_pred_projections[instance_shift_xy], pred_value);
        atomicAdd(&xz_pred_projections[instance_shift_xz], pred_value);
        atomicAdd(&yz_pred_projections[instance_shift_yz], pred_value);

        // Add the targets to the corresponding views
        atomicAdd(&xy_target_projections[instance_shift_xy], target_value);
        atomicAdd(&xz_target_projections[instance_shift_xz], target_value);
        atomicAdd(&yz_target_projections[instance_shift_yz], target_value);
    }
}


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
                                     at::Tensor yz_projection_nums){


    // Check inputs
    CHECK_INPUT(s_coords);  // num_voxels x 4 (first for batch, min is at zero for all dims)
    CHECK_INPUT(s_predictions); // num_voxels x num_instances (sigmoid confidences)
    CHECK_INPUT(s_targets); // num_voxels x num_instances (1.0 / 0.0 for supervised, potentially soft labels
    CHECK_INPUT(xy_pred_projections);  // x_dim x y_dim x instance_num (to be filled with sum of values in pred)
    CHECK_INPUT(xz_pred_projections);  // x_dim x z_dim x instance_num (to be filled with sum of values in pred)
    CHECK_INPUT(yz_pred_projections);  // y_dim x z_dim x instance_num (to be filled with sum of values in pred)
    CHECK_INPUT(xy_target_projections);  // x_dim x y_dim x instance_num (to be filled with sum of values in target)
    CHECK_INPUT(xz_target_projections);  // x_dim x z_dim x instance_num (to be filled with sum of values in target)
    CHECK_INPUT(yz_target_projections);  // y_dim x z_dim x instance_num (to be filled with sum of values in target)
    CHECK_INPUT(xy_projection_nums);  // x_dim x y_dim x instance_num (to be filled with number of initialized voxels)
    CHECK_INPUT(xz_projection_nums);  // x_dim x z_dim x instance_num (to be filled with number of initialized voxels)
    CHECK_INPUT(yz_projection_nums);  // y_dim x z_dim x instance_num (to be filled with number of initialized voxels)

    // Get dimensions
    const int voxel_num = s_coords.size(0);
    const int instance_num = s_predictions.size(1);
    const int x_dim = xy_pred_projections.size(0);
    const int y_dim = xy_pred_projections.size(1);
    const int z_dim = xz_pred_projections.size(1);

    const dim3 gridSize((voxel_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    project_sparse_voxels_to_planes_kernel<<<gridSize, blockSize>>>(s_coords.data<int>(),
                                                                    s_predictions.data<float>(),
                                                                    s_targets.data<float>(),
                                                                    xy_pred_projections.data<float>(),
                                                                    xz_pred_projections.data<float>(),
                                                                    yz_pred_projections.data<float>(),
                                                                    xy_target_projections.data<float>(),
                                                                    xz_target_projections.data<float>(),
                                                                    yz_target_projections.data<float>(),
                                                                    xy_projection_nums.data<int>(),
                                                                    xz_projection_nums.data<int>(),
                                                                    yz_projection_nums.data<int>(),
                                                                    voxel_num, instance_num,
                                                                    x_dim, y_dim, z_dim);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif
}


void __global__ project_sparse_voxels_to_planes_backward_kernel(const int* s_coords,
                                                                float* s_grads,
                                                                const float* xy_grads,
                                                                const float* xz_grads,
                                                                const float* yz_grads,
                                                                const int* xy_grad_nums,
                                                                const int* xz_grad_nums,
                                                                const int* yz_grad_nums,
                                                                const int voxel_num, const int inst_num,
                                                                const int x_dim, const int y_dim, const int z_dim){

    //Get voxel index and position
    const unsigned int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;

    // return for invalid voxels
    if (voxel_id >= voxel_num) { return; }

    const int coord_x = s_coords[voxel_id * 4 + 1];
    const int coord_y = s_coords[voxel_id * 4 + 2];
    const int coord_z = s_coords[voxel_id * 4 + 3];

    // This should be an error - views not correctly initialized
    if (coord_x >= x_dim || coord_y >= y_dim || coord_z >= z_dim || coord_x  < 0 || coord_y  < 0 || coord_z < 0) { return; }

    // We don't have instance dimension in nums as that is the same for all of those
    const unsigned int vox_num_xy = coord_x * y_dim + coord_y;
    const unsigned int vox_num_xz = coord_x * z_dim + coord_z;
    const unsigned int vox_num_yz = coord_y * z_dim + coord_z;

    // For valid voxels iterate over all instance predictions/targets
    for (int pred_id = 0; pred_id < inst_num; pred_id++) {

        // Get the array memory locations with coordinate index and instance num
        const unsigned int voxel_grad_id = voxel_id * inst_num + pred_id;
        const unsigned int grad_id_xy = coord_x * y_dim * inst_num + coord_y * inst_num + pred_id;
        const unsigned int grad_id_xz = coord_x * z_dim * inst_num + coord_z * inst_num + pred_id;
        const unsigned int grad_id_yz = coord_y * z_dim * inst_num + coord_z * inst_num + pred_id;

        // Copy the grad values
        // We dont have to  normalize it with the number of voxels as a.ready did for the forward pass
        float grad_xy = xy_grads[grad_id_xy];
        float grad_xz = xz_grads[grad_id_xz];
        float grad_yz = yz_grads[grad_id_yz];

        // Check how many grads were not zero and normalize with that number
        // Necessary as grad will have to be normalized with that
        int grad_num = 0;
        if (grad_xy != 0.0f) { grad_num++; }
        if (grad_xz != 0.0f) { grad_num++; }
        if (grad_yz != 0.0f) { grad_num++; }

        // Update the grad value with the average of the 3 views
        if (grad_num > 0) {  s_grads[voxel_grad_id] = (grad_xy + grad_xz + grad_yz) / grad_num; }
        else {
            // printf("Grad num was zero for (%i, %i, %i)\n", coord_x, coord_y, coord_z);
            s_grads[voxel_grad_id] = 0.0f;

        }

    }
}


void project_sparse_voxels_to_planes_backward(at::Tensor s_coords,
                                              at::Tensor s_grads,
                                              at::Tensor xy_grads,
                                              at::Tensor xz_grads,
                                              at::Tensor yz_grads,
                                              at::Tensor xy_nums,
                                              at::Tensor xz_nums,
                                              at::Tensor yz_nums){

    // Check inputs - shapes are the same as in forward pass
    CHECK_INPUT(s_coords);
    CHECK_INPUT(s_grads);
    CHECK_INPUT(xy_grads);  // these are the 2D gradients that we got with some sort of BCE
    CHECK_INPUT(xz_grads);
    CHECK_INPUT(yz_grads);
    CHECK_INPUT(xy_nums);  // For every pixel how many voxels were projected
    CHECK_INPUT(xz_nums);
    CHECK_INPUT(yz_nums);

    // Get dimensions
    const int voxel_num = s_coords.size(0);
    const int instance_num = xy_grads.size(2);
    const int x_dim = xy_grads.size(0);
    const int y_dim = xy_grads.size(1);
    const int z_dim = xz_grads.size(1);

    const dim3 gridSize((voxel_num + T_PER_BLOCK - 1) / T_PER_BLOCK, 1, 1);
    const dim3 blockSize(T_PER_BLOCK, 1, 1);

    project_sparse_voxels_to_planes_backward_kernel<<<gridSize, blockSize>>>(s_coords.data<int>(),
                                                                             s_grads.data<float>(),
                                                                             xy_grads.data<float>(),
                                                                             xz_grads.data<float>(),
                                                                             yz_grads.data<float>(),
                                                                             xy_nums.data<int>(),
                                                                             xz_nums.data<int>(),
                                                                             yz_nums.data<int>(),
                                                                             voxel_num, instance_num,
                                                                             x_dim, y_dim, z_dim);

#ifdef _DEBUG
    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg(__FUNCTION__);
#endif

}