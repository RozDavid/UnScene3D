"""
Author: David Rozenberszki (david.rozenberszki@tum.de)
Date: Dec 21, 2022
"""

import torch
from torch import nn
from torch.autograd import Function
from MinkowskiEngine import SparseTensor

import custom_cuda_utils

"""
    Projection losses for noise robust training
"""
class ProjectionFunction(Function):
    @staticmethod
    def forward(ctx, s_coords, s_predictions, s_targets,
                xy_pred_projections, xz_pred_projections, yz_pred_projections,
                xy_target_projections, xz_target_projections, yz_target_projections,
                xy_projection_nums, xz_projection_nums, yz_projection_nums):

        # Init and save grad tensor for backward
        grad_shape = s_predictions.shape
        s_grads = torch.zeros(grad_shape, device=s_coords.device)

        # Do the interpolation function with the CUDA API
        custom_cuda_utils.project_sparse_voxels_to_planes(s_coords, s_predictions, s_targets,
                                                          xy_pred_projections, xz_pred_projections, yz_pred_projections,
                                                          xy_target_projections, xz_target_projections, yz_target_projections,
                                                          xy_projection_nums, xz_projection_nums, yz_projection_nums)

        # Save for backward pass
        variables = [s_coords, s_grads, xy_projection_nums, xz_projection_nums, yz_projection_nums]
        ctx.save_for_backward(*variables)

        # Normalize the projections with the number of associated voxels
        xy_pred_projections = xy_pred_projections / (xy_projection_nums.unsqueeze(-1) + 10e-9)
        xz_pred_projections = xz_pred_projections / (xz_projection_nums.unsqueeze(-1) + 10e-9)
        yz_pred_projections = yz_pred_projections / (yz_projection_nums.unsqueeze(-1) + 10e-9)

        # Normalize the targets too
        xy_target_projections = xy_target_projections / (xy_projection_nums.unsqueeze(-1) + 10e-9)
        xz_target_projections = xz_target_projections / (xz_projection_nums.unsqueeze(-1) + 10e-9)
        yz_target_projections = yz_target_projections / (yz_projection_nums.unsqueeze(-1) + 10e-9)

        # Filter out NaNs
        xy_target_projections[xy_projection_nums == 0] = 0.
        xz_target_projections[xz_projection_nums == 0] = 0.
        yz_target_projections[yz_projection_nums == 0] = 0.
        xy_pred_projections[xy_projection_nums == 0] = 0.
        xz_pred_projections[xz_projection_nums == 0] = 0.
        yz_pred_projections[yz_projection_nums == 0] = 0.

        # Return all necessary components
        return xy_pred_projections, xz_pred_projections, yz_pred_projections, xy_target_projections, xz_target_projections, yz_target_projections, xy_projection_nums, xz_projection_nums, yz_projection_nums

    @staticmethod
    def backward(ctx, xy_pred_projections_grad, xz_pred_projections_grad, yz_pred_projections_grad,
                 xy_target_projections_grad, xz_target_projections_grad, yz_target_projections_grad,
                 xy_projection_nums_grad, xz_projection_nums_grad, yz_projection_nums_grad):

        # Retrieve forward params
        s_coords, s_grads, xy_pred_projection_nums, xz_pred_projection_nums, yz_pred_projection_nums = ctx.saved_variables

        # Backward pass is essentially a weighted averaging of sampled point gradients
        custom_cuda_utils.project_sparse_voxels_to_planes_backward(s_coords, s_grads,
                                                                   xy_pred_projections_grad.contiguous(), xz_pred_projections_grad.contiguous(), yz_pred_projections_grad.contiguous(),
                                                                   xy_pred_projection_nums, xz_pred_projection_nums, yz_pred_projection_nums)
        # Only return grad for sparse predictions
        return None, s_grads, None, None, None, None, None, None, None, None, None, None, None, None, None


class ProjectionFunctionWrapper(nn.Module):
    def __init__(self):
        super(ProjectionFunctionWrapper, self).__init__()

    def forward(self, s_coords, s_predictions, s_targets):

        # Center coordinates for dense predictions
        shift = torch.amin(s_coords, 0)
        centered_coords = s_coords - shift

        x_dim, y_dim, z_dim = centered_coords[:, 1:].max(0)[0]
        num_instances = s_predictions.shape[1]
        xy_pred_projections = torch.zeros((x_dim, y_dim, num_instances), device=s_coords.device)
        xz_pred_projections = torch.zeros((x_dim, z_dim, num_instances), device=s_coords.device)
        yz_pred_projections = torch.zeros((y_dim, z_dim, num_instances), device=s_coords.device)

        xy_target_projections = torch.zeros((x_dim, y_dim, num_instances), device=s_coords.device)
        xz_target_projections = torch.zeros((x_dim, z_dim, num_instances), device=s_coords.device)
        yz_target_projections = torch.zeros((y_dim, z_dim, num_instances), device=s_coords.device)

        xy_projection_nums = torch.zeros((x_dim, y_dim), device=s_coords.device, dtype=torch.int)
        xz_projection_nums = torch.zeros((x_dim, z_dim), device=s_coords.device, dtype=torch.int)
        yz_projection_nums = torch.zeros((y_dim, z_dim), device=s_coords.device, dtype=torch.int)

        # Calculate projection with autograd
        return ProjectionFunction.apply(centered_coords.int().contiguous(), s_predictions.contiguous(), s_targets.contiguous(),
                                        xy_pred_projections, xz_pred_projections, yz_pred_projections,
                                        xy_target_projections, xz_target_projections, yz_target_projections,
                                        xy_projection_nums, xz_projection_nums, yz_projection_nums), (x_dim, y_dim, z_dim)


class ProjectionMaskLoss(nn.Module):

    def __init__(self, config=None, base_loss='bce', directions='xyz'):
        super(ProjectionMaskLoss, self).__init__()

        self.base_loss = base_loss
        self.eps = 10e-9
        self.directions = directions

        # self.target_projection_mode = self.config.SOLO3D.projection_loss.target_projection_mode

        # use the wrapped autograd the cuda projection module
        self.projection_module = ProjectionFunctionWrapper()

        criterion = None
        if self.base_loss == 'bce':
            criterion = nn.BCELoss(reduction='none')
        else:
            raise NotImplementedError
        self.criterion = criterion

    def forward(self, all_mask_preds, all_mask_targets, coords):
        """
        :param all_mask_preds: Dense mask predictions by all grids in a single batch
        :param all_mask_targets: Dense mask targets for all grids in a single batch
        :param coords: The original coordinates used for SparseTensors - to be used for generating dense voxels
        :return: reduced loss value with supervised pseudo images as targets/preds
        """

        # Use the projection module to get views
        inst_num, point_num = all_mask_preds.shape
        proj_outputs, dims = self.projection_module(coords, torch.sigmoid(all_mask_preds.T), all_mask_targets.T)
        xy_pred_proj, xz_pred_proj, yz_pred_proj, xy_target_proj, xz_target_proj, yz_target_proj, xy_projection_nums, xz_projection_nums, yz_projection_nums = proj_outputs
        x_dim, y_dim, z_dim = dims
        xy_target_proj, xz_target_proj, yz_target_proj = xy_target_proj.detach(), xz_target_proj.detach(), yz_target_proj.detach()

        # get all shapes to determine normalization
        all_shape = inst_num * (len(xy_projection_nums.nonzero()) + len(xz_projection_nums.nonzero()) + len(yz_projection_nums.nonzero()))

        # Add losses for all requested directions
        # Only keep the items, where there is projected value
        loss = 0
        if 'x' in self.directions:
            tmp_loss = self.criterion(yz_pred_proj, yz_target_proj)
            tmp_loss[yz_projection_nums == 0] = 0.
            loss += tmp_loss.sum()
        if 'y' in self.directions:
            tmp_loss = self.criterion(xz_pred_proj, xz_target_proj)
            tmp_loss[xz_projection_nums == 0] = 0.
            loss += tmp_loss.sum()
        if 'z' in self.directions:
            tmp_loss = self.criterion(xy_pred_proj, xy_target_proj)
            tmp_loss[xy_projection_nums == 0] = 0.
            loss += tmp_loss.sum()

        if torch.any(torch.isnan(loss)) or all_shape == 0:
            debug = 0

        return loss, all_shape
