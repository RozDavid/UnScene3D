import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch import nn
from torch.autograd import Function
from MinkowskiEngine import SparseTensor

import raycast_cuda
import project_features_cuda

'''
2D --> 3D
Take the precomputed mappings from the dataloader
and use those to iterate over all voxels and copy the features
'''

class Project2DFeaturesCUDA(nn.Module):
    def __init__(self, width, height, voxel_size, config, depth_min=0.1, depth_max=4.0):
        super(Project2DFeaturesCUDA, self).__init__()

        self.image_width = width
        self.image_height = height
        self.voxel_size = voxel_size
        self.ray_increment = voxel_size / 2.
        self.config = config
        self.depth_min = depth_min / voxel_size
        self.depth_max = depth_max / voxel_size

    def forward(self, encoded_2d_features, coords, view_matrix, intrinsic_params, pred_mode=False):

        # Create sparse indexer array
        voxel_num = coords.shape[0]
        batch_size = coords[-1, 0] + 1

        # Shift coordinates and poses for cuda indexing
        shifts = []
        local_coords = coords.detach().clone()
        local_views = view_matrix.detach().clone()
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            shift = torch.amin(coords[batch_mask, 1:], 0)
            shifts += [shift]
            local_coords[batch_mask, 1:] = local_coords[batch_mask, 1:] - shift

            # Update 2D views
            local_views[b, :, :3, 3] -= shift


        occupancies_3d = SparseTensor(features=torch.arange(voxel_num, device=coords.device).view(-1, 1),
                                      coordinates=local_coords,
                                      device=local_coords.device)

        occupancies_3d = occupancies_3d.dense()[0].long().permute(0, 1, 4, 3, 2).contiguous().squeeze(1)  # (batch, z, y, x)

        # Initialize CUDA tensors
        mapping2dto3d_num = torch.zeros(voxel_num, dtype=torch.int, device=occupancies_3d.device)

        # Combine Raycast params in one tensor
        opts = torch.FloatTensor([self.image_width, self.image_height, self.depth_min, self.depth_max, self.ray_increment])

        ## make feature tensor contiguous
        encoded_2d_features = encoded_2d_features if encoded_2d_features.is_contiguous() else encoded_2d_features.contiguous()

        if not pred_mode:  # this averages the features
            projected_features = torch.zeros((voxel_num, encoded_2d_features.shape[-1]), dtype=encoded_2d_features.dtype, device=occupancies_3d.device)

            # Project all and normalize projected features with number of added voxels
            project_features_cuda.project_features_cuda(encoded_2d_features, occupancies_3d, local_views, intrinsic_params, opts, mapping2dto3d_num, projected_features, torch.BoolTensor([pred_mode]))
            projected_features = torch.div(projected_features, mapping2dto3d_num.view(-1, 1) + 10e-5)

        else:  # this takes the majority element along the last axis
            projected_features = torch.zeros((voxel_num, encoded_2d_features.shape[-1]), dtype=encoded_2d_features.dtype, device=occupancies_3d.device) + self.config.data.ignore_label
            project_features_cuda.project_features_cuda(encoded_2d_features, occupancies_3d, local_views, intrinsic_params, opts, mapping2dto3d_num, projected_features, torch.BoolTensor([pred_mode]))
            projected_features = projected_features.flatten().long()

        return projected_features, mapping2dto3d_num


'''
3D --> 2D
Render out features to query camera poses from the features voxel grid
'''
class RayCastFeaturesFunction(Function):
    @staticmethod
    def forward(ctx, occ3d, features_3d, indexes_image, view_matrix, intrinsic_params, opts, mapping3dto2d_num, ignore_label):

        raycast_cuda.raycast_features(occ3d, indexes_image, view_matrix, intrinsic_params, opts, mapping3dto2d_num)

        # Index out the features from the 3D voxels to 2D
        projected_features = torch.zeros((indexes_image.flatten().shape[0], features_3d.shape[-1]), dtype=torch.float, device=indexes_image.device)
        matched_inds = indexes_image.flatten() != ignore_label
        projected_features[matched_inds] = features_3d[indexes_image.flatten()[matched_inds]]
        projected_features[indexes_image.flatten() == ignore_label] = 0.

        # Storing the derivative on the voxels
        d_voxels = torch.zeros(features_3d.shape, device=features_3d.device).contiguous()

        variables = [d_voxels, occ3d, indexes_image, mapping3dto2d_num]
        ctx.save_for_backward(*variables)

        return projected_features, indexes_image, mapping3dto2d_num

    @staticmethod
    def backward(ctx, grad_image, grad_indexes_image, grad_mapping_nums):

        d_voxels, occ3d, indexes_image, mapping3dto2d_num = ctx.saved_variables

        raycast_cuda.raycast_features_backward(occ3d.contiguous(), indexes_image.contiguous(), grad_image.contiguous(), d_voxels.contiguous())

        # Normalize grad_values with number of associated pixels
        d_voxels = torch.div(d_voxels, mapping3dto2d_num.view(-1, 1) + 10e-5)

        # print(grad_image.abs().sum(), grad_indexes_image.sum(), grad_mapping_nums.sum())
        # print(f'Backward called with grad image of shape: {grad_image.shape} and min/max value of {grad_image.min():.4f}/{grad_image.max():.4f}')
        # print(f'Gradients calculated for voxels with: {d_voxels.shape} and min/max value of {d_voxels.min():.10f}/{d_voxels.max():.10f}')
        # print('mapping3dto2d_num:', mapping3dto2d_num.shape, mapping3dto2d_num.max())

        return None, d_voxels, None, None, None, None, None, None

    @staticmethod
    def backward_debug(occ3d, indexes_image, grad_image, d_voxels, mapping3dto2d_num):
        raycast_cuda.raycast_features_backward(occ3d, indexes_image, grad_image, d_voxels)
        d_voxels = torch.div(d_voxels, mapping3dto2d_num.view(-1, 1) + 10e-5)
        return d_voxels


class RaycastFeatures(nn.Module):
    def __init__(self, dims3d, width, height, voxel_size, config, depth_min=0.1, depth_max=4.0):
        super(RaycastFeatures, self).__init__()
        self.dims3d = dims3d
        self.width = width
        self.height = height
        self.voxel_size = voxel_size
        self.ray_increment = voxel_size / 2.
        self.depth_min = depth_min / voxel_size
        self.depth_max = depth_max / voxel_size
        self.config = config

    def forward(self, s_3d_output, view_matrix, intrinsic_params):

        # Get batch size and allocate indexes image
        batch_size = s_3d_output.C[-1, 0] + 1
        indexes_image = torch.zeros(batch_size, view_matrix.shape[1], self.height, self.width, dtype=torch.long, device=s_3d_output.C.device).contiguous() + self.config.data.ignore_label

        # Create sparse indexer array
        voxel_num = s_3d_output.C.shape[0]

        # Shift coordinates and poses for cuda indexing
        coords = s_3d_output.C.detach().clone().contiguous()
        local_views = view_matrix.detach().clone()
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            shift = torch.amin(coords[batch_mask, 1:], 0)
            coords[batch_mask, 1:] = coords[batch_mask, 1:] - shift

            # Update 2D views
            local_views[b, :, :3, 3] -= shift

        occ3d = SparseTensor(features=torch.arange(voxel_num, device=s_3d_output.C.device).view(-1, 1),
                             coordinates=coords,
                             device=s_3d_output.C.device)

        occ3d = occ3d.dense()[0].long().permute(0, 1, 4, 3, 2).squeeze(1).contiguous()  # (batch, z, y, x)
        # self.occ3d = occ3d # for debugging only

        # Storing the number of associated pixels for every voxel
        mapping3dto2d_num = torch.zeros(voxel_num, dtype=torch.int, device=occ3d.device).contiguous()

        # Combine Raycast params in one tensor
        opts = torch.FloatTensor([self.width, self.height, self.depth_min, self.depth_max, self.ray_increment, self.dims3d[2], self.dims3d[1], self.dims3d[0]])

        # Calculate forward raycasting with grading wih autograd attach
        return RayCastFeaturesFunction.apply(occ3d, s_3d_output.F.contiguous(), indexes_image, local_views, intrinsic_params, opts, mapping3dto2d_num, self.config.data.ignore_label)


'''
With Trilinear Interpolation
'''
class RayCastInterpolateFunction(Function):
    @staticmethod
    def forward(ctx,
                features_3d,
                projected_features,
                occ3d,
                indexes_image,
                vox_dist_weights,
                view_matrix, intrinsic_params, opts, mapping3dto2d_num):


        raycast_cuda.raycast_interpolate_features(features_3d,
                                                  projected_features,
                                                  occ3d,
                                                  indexes_image,
                                                  vox_dist_weights,
                                                  view_matrix,
                                                  intrinsic_params,
                                                  opts,
                                                  mapping3dto2d_num)

        # Storing the derivative on the voxels
        d_voxels = torch.zeros(features_3d.shape, device=features_3d.device).contiguous()

        # Save backward variables
        variables = [d_voxels, occ3d, indexes_image, vox_dist_weights, mapping3dto2d_num]
        ctx.save_for_backward(*variables)

        return projected_features, indexes_image, vox_dist_weights, mapping3dto2d_num

    @staticmethod
    def backward(ctx, grad_image, grad_indexes_image, grad_vox_dist_weights, grad_mapping_nums):

        d_voxels, occ3d, indexes_image, vox_dist_weights, mapping3dto2d_num = ctx.saved_variables

        raycast_cuda.raycast_interpolate_backward(occ3d, indexes_image, vox_dist_weights, grad_image.contiguous(), d_voxels)

        # Normalize grad_values with number of associated pixels
        d_voxels = torch.div(d_voxels, mapping3dto2d_num.view(-1, 1) + 10e-5)

        # print(f'Grad image shape was {grad_image.shape}, while d_voxels {d_voxels.shape}')

        return d_voxels, None, None, None, None, None, None, None, None

    @staticmethod
    def backward_debug(occ3d, indexes_image, vox_dist_weights, grad_image, d_voxels, mapping3dto2d_num):
        raycast_cuda.raycast_interpolate_backward(occ3d, indexes_image, vox_dist_weights, grad_image.contiguous(), d_voxels)
        d_voxels = torch.div(d_voxels, mapping3dto2d_num.view(-1, 1) + 10e-5)
        return d_voxels

class RaycastInterpolateFeatures(RaycastFeatures):
    def __init__(self, dims3d, width, height, voxel_size, config, depth_min=0.1, depth_max=4.0):
        super().__init__(dims3d, width, height, voxel_size, config)

    def forward(self, s_3d_output, view_matrix, intrinsic_params):

        # Get batch size and allocate interpolate indexes image and tensor storing weights
        batch_size = s_3d_output.C[-1, 0] + 1
        voxel_num = s_3d_output.C.shape[0]
        view_num = view_matrix.shape[1]
        indexes_image = torch.zeros((batch_size, view_num, self.height, self.width, 8), dtype=torch.long, device=s_3d_output.C.device).contiguous() + self.config.data.ignore_label
        vox_dist_weights = torch.zeros((batch_size, view_num, self.height, self.width, 8), device=s_3d_output.C.device).contiguous()

        # Storing the number of associated pixels for every voxel
        mapping3dto2d_num = torch.zeros(voxel_num, dtype=torch.float, device=s_3d_output.C.device).contiguous()

        # Create output projected features tensor
        projected_features = torch.zeros((batch_size, view_num, self.height, self.width, s_3d_output.F.shape[-1]),
                                         dtype=torch.float,
                                         device=indexes_image.device).contiguous()

        # Shift coordinates and poses for cuda indexing
        coords = s_3d_output.C.detach().clone().contiguous()
        local_views = view_matrix.detach().clone().contiguous()
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            shift = torch.amin(coords[batch_mask, 1:], 0)
            coords[batch_mask, 1:] = coords[batch_mask, 1:] - shift

            # Update 2D views
            local_views[b, :, :3, 3] -= shift

        # Create dense tensor
        occ3d = SparseTensor(features=torch.arange(voxel_num, device=s_3d_output.C.device).view(-1, 1),
                             coordinates=coords,
                             device=s_3d_output.C.device)
        occ3d = occ3d.dense()[0].long().permute(0, 1, 4, 3, 2).squeeze(1).contiguous()  # (batch, z, y, x)
        # self.occ3d = occ3d  # for debugging only

        # Combine Raycast params in one tensor
        opts = torch.FloatTensor([self.width, self.height, self.depth_min, self.depth_max, self.ray_increment, self.dims3d[2], self.dims3d[1], self.dims3d[0]])

        # Do interpolated feature projection with backward registration
        return RayCastInterpolateFunction.apply(s_3d_output.F.contiguous(), projected_features,
                                                occ3d, indexes_image, vox_dist_weights,
                                                local_views, intrinsic_params, opts, mapping3dto2d_num)
