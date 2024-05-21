import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import matplotlib
import torch
import open3d as o3d

import MinkowskiEngine as ME
import torchvision.transforms as transforms


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords, feats, indexes):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)

        return coords, feats, indexes


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords, feats, indexes):
        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)
            hi = feats[:, :3].max(0, keepdims=True)
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / ((hi - lo) + 1.)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats

        return coords, feats, indexes


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, coords, feats, indexes):
        if random.random() < 0.95:
            noise = np.random.randn(feats.shape[0], 3)
            noise *= self.std * 255
            feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)

        return coords, feats, indexes


class ChromaticScale(object):

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def __call__(self, coords, feats, indexes):

        feats[:, :3] = feats[:, :3] * self.scale_factor

        return coords, feats, indexes


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, indexes, corrs=None):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, indexes


##############################
# Coordinate transformations
##############################
class RandomDropout(object):

    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, coords, feats, indexes):
        if random.random() < self.dropout_ratio:
            N = len(coords)
            inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)

            coords = coords[inds]
            feats = feats[inds]
            indexes = indexes[inds]

        return coords, feats, indexes


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, indexes):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]

        return coords, feats, indexes


class ElasticDistortion:

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, indexes, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords, feats, indexes

    def __call__(self, coords, feats, indexes):
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, indexes = self.elastic_distortion(coords, feats, indexes, granularity, magnitude)

        return coords, feats, indexes


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

class Tranform2D(object):

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize([480, 640]),
            ]
        )

    def __call__(self, batched_images):
        return self.transform(batched_images)


class cfl_collate_fn_factory:
    """Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, instances, scene_names, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations = list(zip(*list_data))
        coords_batch, feats_batch, labels_batch, instances_batch, scene_names_batch, images_batch, camera_poses_batch, color_intrinsics_batch, segment_ids_batch, seg_connectivity_batch, transformations_batch = [], [], [], [], [], [], [], [], [], [], []

        batch_id = 0
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                logging.warning(
                    f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
                    f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
                )
                break
            coords_batch.append(torch.from_numpy(coords[batch_id]))
            feats_batch.append(torch.from_numpy(feats[batch_id]))
            labels_batch.append(torch.from_numpy(labels[batch_id]).long())
            instances_batch.append(torch.from_numpy(instances[batch_id]).long())
            scene_names_batch.append(scene_names[batch_id])
            transformations_batch.append(torch.from_numpy(transformations[batch_id]))

            if images[batch_id] is not None:
                images_batch.append(images[batch_id].unsqueeze(0))
                camera_poses_batch = np.append(camera_poses_batch, camera_poses[batch_id][np.newaxis, :], axis=0) if camera_poses_batch != [] else camera_poses[batch_id][np.newaxis, :]
                color_intrinsics_batch = np.append(color_intrinsics_batch, color_intrinsics[batch_id][np.newaxis, :], axis=0) if color_intrinsics_batch != [] else color_intrinsics[batch_id][np.newaxis, :]

            if segment_ids[batch_id] is not None:
                # append segments ids with collated batch id
                curr_seg_ids = segment_ids[batch_id]
                curr_connectivity = seg_connectivity[batch_id]
                segment_ids_batch.append(curr_seg_ids)
                seg_connectivity_batch.append(curr_connectivity)

            batch_id += 1

        transformations_batch = torch.stack(transformations_batch).float()

        if images[0] is not None:
            images_batch = torch.cat(images_batch, axis=0)
            camera_poses_batch = torch.from_numpy(camera_poses_batch).float()
            color_intrinsics_batch = torch.from_numpy(color_intrinsics_batch).float()

        if segment_ids[0] is not None:
            segment_ids_batch = torch.from_numpy(np.concatenate(segment_ids_batch, axis=0)).long()

        # Concatenate all lists
        batched_coords, batched_feats, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
        _, _, batched_instances = ME.utils.sparse_collate(coords_batch, feats_batch, instances_batch)
        return batched_coords, batched_feats.float(), labels_batch.long(), batched_instances.long(), scene_names_batch, images_batch, camera_poses_batch, color_intrinsics_batch, segment_ids_batch, seg_connectivity_batch, transformations_batch


# For SOLO3D processing - focusing on instance objects for supervision and also grid associations
class cfl_instance_collate_fn_factory:

    def __init__(self, limit_numpoints):
        self.limit_numpoints = limit_numpoints

    def __call__(self, list_data):
        coords, feats, labels, instances, scene_names, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations, *instance_info = list(zip(*list_data))
        cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints)
        coords_batch, feats_batch, labels_batch, instances_batch, scene_names_batch, images_batch, camera_poses_batch, color_intrinsics_batch, segment_ids_batch, seg_connectivity_batch, transformations_batch = cfl_collate_fn(list(zip(coords, feats, labels, instances, scene_names, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations)))
        num_truncated_batch = coords_batch[-1, 0].item() + 1

        cloud_instances, grid_locations_dict, grid_indices = instance_info
        cloud_instances_batch, grid_locations_dict_batch, grid_indices_batch = [], [], []

        for batch_id in range(num_truncated_batch):
            cloud_instances_batch += [cloud_instances[batch_id]]
            grid_locations_dict_batch += [grid_locations_dict[batch_id]]
            grid_indices_batch += [torch.from_numpy(grid_indices[batch_id])]

        grid_indices_batch = torch.cat(grid_indices_batch).long()

        return coords_batch, feats_batch, labels_batch, instances_batch, scene_names_batch, images_batch, camera_poses_batch, color_intrinsics_batch, segment_ids_batch, seg_connectivity_batch, transformations_batch, cloud_instances_batch, grid_locations_dict_batch, grid_indices_batch