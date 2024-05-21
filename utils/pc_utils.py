import os

import json
import numpy as np
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from PIL import Image, ImageOps


from scipy.spatial import KDTree

from constants.scannet_constants import *


def read_plyfile(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True, with_refinement=False, faces=None):
    """Save an RGB point cloud as a PLY file.

    Args:
      points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
          the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
    """
    assert points_3d.ndim == 2
    if with_label:
        if points_3d.shape[1] == 7:
            python_types = (float, float, float, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4')]

        if points_3d.shape[1] == 8:
            python_types = (float, float, float, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4'), ('instance_id', 'u4')]

        # Save with segment_id
        elif points_3d.shape[1] == 9:
            python_types = (float, float, float, int, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('label', 'u4'), ('instance_id', 'u4'), ('segment_id', 'u4')]

        # Save with multiple labels
        elif points_3d.shape[1] == 11:
            python_types = (float, float, float, int, int, int, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('supercat_label', 'u4'), ('parent_label', 'u4'), ('child_label', 'u4'),
                         ('supercat_refinement', 'u4'), ('parent_refinement', 'u4')]

    else:
        if points_3d.shape[1] == 3:
            gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
            points_3d = np.hstack((points_3d, gray_concat))
        elif points_3d.shape[1] == 6:
            python_types = (float, float, float, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1')]
        else:
            pass


    if with_refinement:

        if points_3d.shape[1] == 7:
            python_types = (float, float, float, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('refinement', 'u4')]
        #Save with both refinement
        elif points_3d.shape[1] == 8:
            python_types = (float, float, float, int, int, int, int, int)
            npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                         ('blue', 'u1'), ('supercat_refinement', 'u4'), ('parent_refinement', 'u4')]
        else:
            pass


    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        elements = [PlyElement.describe(vertices_array, 'vertex')]

        if faces is not None:
            faces_array = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
            faces_array['vertex_indices'] = faces
            elements += [PlyElement.describe(faces_array, 'face')]

        # Write
        PlyData(elements).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'property uchar alpha\n'
                    'end_header\n' % points_3d.shape[0])
            for row_idx in range(points_3d.shape[0]):
                X, Y, Z, R, G, B = points_3d[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)


class Camera(object):

    def __init__(self, intrinsics):
        self._intrinsics = intrinsics
        self._camera_matrix = self.build_camera_matrix(self.intrinsics)
        self._K_inv = inv(self.camera_matrix)

    @staticmethod
    def build_camera_matrix(intrinsics):
        """Build the 3x3 camera matrix K using the given intrinsics.

        Equation 6.10 from HZ.
        """
        f = intrinsics['focal_length']
        pp_x = intrinsics['pp_x']
        pp_y = intrinsics['pp_y']

        K = np.array([[f, 0, pp_x], [0, f, pp_y], [0, 0, 1]], dtype=np.float32)
        # K[:, 0] *= -1.  # Step 1 of Kyle
        assert matrix_rank(K) == 3
        return K

    @staticmethod
    def extrinsics2RT(extrinsics):
        """Convert extrinsics matrix to separate rotation matrix R and translation vector T.
        """
        assert extrinsics.shape == (4, 4)
        R = extrinsics[:3, :3]
        T = extrinsics[3, :3]
        R = np.copy(R)
        T = np.copy(T)
        T = T.reshape(3, 1)
        R[0, :] *= -1.  # Step 1 of Kyle
        T *= 100.  # Convert from m to cm
        return R, T

    def project(self, points_3d, extrinsics=None):
        """Project a 3D point in camera coordinates into the camera/image plane.

        Args:
          point_3d:
        """
        if extrinsics is not None:  # Map points to camera coordinates
            points_3d = self.world2camera(extrinsics, points_3d)

        # Make sure to handle homogeneous AND non-homogeneous coordinate points
        # Consider handling a set of points
        raise NotImplementedError

    def backproject(self,
                    depth_map,
                    labels=None,
                    max_depth=None,
                    max_height=None,
                    min_height=None,
                    rgb_img=None,
                    extrinsics=None,
                    prune=True):
        """Backproject a depth map into 3D points (camera coordinate system). Attach color if RGB image
        is provided, otherwise use gray [128 128 128] color.

        Does not show points at Z = 0 or maximum Z = 65535 depth.

        Args:
          labels: Tensor with the same shape as depth map (but can be 1-channel or 3-channel).
          max_depth: Maximum depth in cm. All pts with depth greater than max_depth will be ignored.
          max_height: Maximum height in cm. All pts with height greater than max_height will be ignored.

        Returns:
          points_3d: Numpy array of size Nx3 (XYZ) or Nx6 (XYZRGB).
        """
        if labels is not None:
            assert depth_map.shape[:2] == labels.shape[:2]
            if (labels.ndim == 2) or ((labels.ndim == 3) and (labels.shape[2] == 1)):
                n_label_channels = 1
            elif (labels.ndim == 3) and (labels.shape[2] == 3):
                n_label_channels = 3

        if rgb_img is not None:
            assert depth_map.shape[:2] == rgb_img.shape[:2]
        else:
            rgb_img = np.ones_like(depth_map, dtype=np.uint8) * 255

        # Convert from 1-channel to 3-channel
        if (rgb_img.ndim == 3) and (rgb_img.shape[2] == 1):
            rgb_img = np.tile(rgb_img, [1, 1, 3])

        # Convert depth map to single channel if it is multichannel
        if (depth_map.ndim == 3) and depth_map.shape[2] == 3:
            depth_map = np.squeeze(depth_map[:, :, 0])
        depth_map = depth_map.astype(np.float32)

        # Get image dimensions
        H, W = depth_map.shape

        # Create meshgrid (pixel coordinates)
        Z = depth_map
        A, B = np.meshgrid(range(W), range(H))
        ones = np.ones_like(A)
        grid = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis], ones[:, :, np.newaxis]),
                              axis=2)
        grid = grid.astype(np.float32) * Z[:, :, np.newaxis]
        # Nx3 where each row is (a*Z, b*Z, Z)
        grid_flattened = grid.reshape((-1, 3))
        grid_flattened = grid_flattened.T  # 3xN where each col is (a*Z, b*Z, Z)
        prod = np.dot(self.K_inv, grid_flattened)
        XYZ = np.concatenate((prod[:2, :].T, Z.flatten()[:, np.newaxis]), axis=1)  # Nx3
        XYZRGB = np.hstack((XYZ, rgb_img.reshape((-1, 3))))
        points_3d = XYZRGB

        if labels is not None:
            labels_reshaped = labels.reshape((-1, n_label_channels))

        # Prune points
        if prune is True:
            valid = []
            for idx in range(points_3d.shape[0]):
                cur_y = points_3d[idx, 1]
                cur_z = points_3d[idx, 2]
                if (cur_z == 0) or (cur_z == 65535):  # Don't show things at 0 distance or max distance
                    continue
                elif (max_depth is not None) and (cur_z > max_depth):
                    continue
                elif (max_height is not None) and (cur_y > max_height):
                    continue
                elif (min_height is not None) and (cur_y < min_height):
                    continue
                else:
                    valid.append(idx)
            points_3d = points_3d[np.asarray(valid)]
            if labels is not None:
                labels_reshaped = labels_reshaped[np.asarray(valid)]

        if extrinsics is not None:
            points_3d = self.camera2world(extrinsics, points_3d)

        if labels is not None:
            points_3d_labels = np.hstack((points_3d[:, :3], labels_reshaped))
            return points_3d, points_3d_labels
        else:
            return points_3d

    @staticmethod
    def _camera2world_transform(no_rgb_points_3d, R, T):
        points_3d_world = (np.dot(R.T, no_rgb_points_3d.T) - T).T  # Nx3
        return points_3d_world

    @staticmethod
    def _world2camera_transform(no_rgb_points_3d, R, T):
        points_3d_world = (np.dot(R, no_rgb_points_3d.T + T)).T  # Nx3
        return points_3d_world

    def _transform_points(self, points_3d, extrinsics, transform):
        """Base/wrapper method for transforming points using R and T.
        """
        assert points_3d.ndim == 2
        orig_points_3d = points_3d
        points_3d = np.copy(orig_points_3d)
        if points_3d.shape[1] == 6:  # XYZRGB
            points_3d = points_3d[:, :3]
        elif points_3d.shape[1] == 3:  # XYZ
            points_3d = points_3d
        else:
            raise ValueError('3D points need to be XYZ or XYZRGB.')

        R, T = self.extrinsics2RT(extrinsics)
        points_3d_world = transform(points_3d, R, T)

        # Add color again (if appropriate)
        if orig_points_3d.shape[1] == 6:  # XYZRGB
            points_3d_world = np.hstack((points_3d_world, orig_points_3d[:, -3:]))
        return points_3d_world

    def camera2world(self, extrinsics, points_3d):
        """Transform from camera coordinates (3D) to world coordinates (3D).

        Args:
          points_3d: Nx3 or Nx6 matrix of N points with XYZ or XYZRGB values.
        """
        return self._transform_points(points_3d, extrinsics, self._camera2world_transform)

    def world2camera(self, extrinsics, points_3d):
        """Transform from world coordinates (3D) to camera coordinates (3D).
        """
        return self._transform_points(points_3d, extrinsics, self._world2camera_transform)

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @property
    def K_inv(self):
        return self._K_inv


def colorize_pointcloud(xyz, label, num_labels=200, ignore_label=255):

    if num_labels == 20:
        color_map = SCANNET_COLOR_MAP_20
    elif num_labels == 200:
        color_map = SCANNET_COLOR_MAP_200
    else:
        color_map = SCANNET_COLOR_MAP_LONG

    label_rgb = np.vectorize(lambda i: color_map[i])(label)
    label_rgb = np.array(label_rgb).T

    return np.hstack((xyz, label_rgb))


def point_indices_from_group(points, seg_indices, group, labels_pd, CLASS_IDs, nyu40):

    # get the label id for the whole instance group
    group_segments = np.array(group['segments'])
    label = group['label']

    # get the column name to map label from
    target_colum_name = 'id' if not nyu40 else 'nyu40id'

    # map the label id to the correct form
    label_ids = labels_pd[labels_pd['raw_category'] == label][target_colum_name]
    label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

    # add ignore label if not valid
    if not label_id in CLASS_IDs:
        label_id = 0

    # get points, where segindices (points labelled with segment ids) are in the group segment list
    point_IDs = np.where(np.isin(seg_indices, group_segments))

    return points[point_IDs], point_IDs[0], label_id


def load_labels(point_cloud, segments_file,
                aggregations_file, labels_pd,
                CLASS_IDs = VALID_CLASS_IDS_20,
                return_instances=True):

    nyu40 = False  # which column to map from labels dataframe
    if CLASS_IDs == VALID_CLASS_IDS_20:
        nyu40 = True

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    # Generate new labels
    labelled_pc = np.zeros(point_cloud.shape[0])
    instance_ids = np.zeros(point_cloud.shape[0])
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(point_cloud, seg_indices, group, labels_pd, CLASS_IDs, nyu40)
        labelled_pc[p_inds] = label_id
        instance_ids[p_inds] = group['id']

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    if return_instances:
        return labelled_pc, instance_ids
    else:
        return labelled_pc

class PlyWriter(object):
    POINTCLOUD_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                        ('blue', 'u1')]

    @classmethod
    def read_txt(cls, txtfile):
        # Read txt file and parse its content.
        with open(txtfile) as f:
            pointcloud = [l.split() for l in f]
        # Load point cloud to named numpy array.
        pointcloud = np.array(pointcloud).astype(np.float32)
        assert pointcloud.shape[1] == 6
        xyz = pointcloud[:, :3].astype(np.float32)
        rgb = pointcloud[:, 3:].astype(np.uint8)
        return xyz, rgb

    @staticmethod
    def write_ply(array, filepath):
        ply_el = PlyElement.describe(array, 'vertex')
        target_path, _ = os.path.split(filepath)
        if target_path != '' and not os.path.exists(target_path):
            os.makedirs(target_path)
        PlyData([ply_el]).write(filepath)

    @classmethod
    def write_vertex_only_ply(cls, vertices, filepath):
        # assume that points are N x 3 np array for vertex locations
        color = 255 * np.ones((len(vertices), 3))
        pc_points = np.array([tuple(p) for p in np.concatenate((vertices, color), axis=1)],
                             dtype=cls.POINTCLOUD_DTYPE)
        cls.write_ply(pc_points, filepath)

    @classmethod
    def write_ply_vert_color(cls, vertices, colors, filepath):
        # assume that points are N x 3 np array for vertex locations
        pc_points = np.array([tuple(p) for p in np.concatenate((vertices, colors), axis=1)],
                             dtype=cls.POINTCLOUD_DTYPE)
        cls.write_ply(pc_points, filepath)

    @classmethod
    def concat_label(cls, target, xyz, label):
        subpointcloud = np.concatenate([xyz, label], axis=1)
        subpointcloud = np.array([tuple(l) for l in subpointcloud], dtype=cls.POINTCLOUD_DTYPE)
        return np.concatenate([target, subpointcloud], axis=0)


def assign_image_features_to_point_cloud(batched_cloud, image_features, camera_poses, camera_intrinsics, depth_images, depth_treshold=2.5):
    # reference equation at https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

    batch_size = int(batched_cloud[-1, 0]) + 1
    sampled_point_num = batched_cloud.shape[0]
    feature_dim = image_features.shape[-1]
    view_num = image_features.shape[1]
    height = image_features.shape[2]
    width = image_features.shape[3]
    device = batched_cloud.device

    # Coordinate frame transform from pose to camera coords
    S_img = torch.Tensor([[-1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0]]).to(device)

    # we need these to properly index the image feature tensor
    batch_shift_mult = view_num * height * width
    view_shift_mult = height * width
    height_shift_mult = width
    flat_features = image_features.view(-1, feature_dim)
    flat_depth = depth_images.flatten()

    # To store values
    projected_point_features = torch.zeros((sampled_point_num, view_num, feature_dim), dtype=flat_features.dtype, device=flat_features.device)

    # To index
    sample_indexer = torch.arange(sampled_point_num)

    # Iterate over all batches
    for batch_num in range(batch_size):
        batch_mask = batched_cloud[:, 0] == batch_num
        intrinsics = camera_intrinsics[batch_num].flatten()
        query_coords = batched_cloud[batch_mask, 1:].squeeze()
        hom_query = torch.cat((query_coords, torch.ones((query_coords.shape[0], 1), device=query_coords.device)), dim=1)
        K_img = torch.Tensor([[intrinsics[0], 0, intrinsics[2]],
                              [0, intrinsics[1], intrinsics[3]],
                              [0, 0, 1]]).to(device)

        batched_intrinsics = (K_img @ S_img).unsqueeze(0).expand(view_num, -1, -1)
        query_projection_ids = (torch.bmm(batched_intrinsics, camera_poses[batch_num].inverse()) @ hom_query.T).permute(0, 2, 1)
        image_space_z = - query_projection_ids[:, :, 2]
        query_projection_ids = torch.round((query_projection_ids[:, :, :2] / query_projection_ids[:, :, 2].unsqueeze(-1))).long().contiguous()
        query_projection_ids[:, :, 1] = height - query_projection_ids[:, :, 1]  # invert y indexing

        # iterate over all views to get valid ids with projections within the frustrum
        for v in range(view_num):
            valid_width = torch.logical_and(query_projection_ids[v, :, 0] >= 0, query_projection_ids[v, :, 0] < width)
            valid_height = torch.logical_and(query_projection_ids[v, :, 1] >= 0, query_projection_ids[v, :, 1] < height)
            valid_projections = torch.logical_and(valid_height, valid_width)

            # Calculate shaped coordinates to flat coordinates
            query_projection_flat_inds = (batch_num * batch_shift_mult) + (v * view_shift_mult) + (query_projection_ids[v, valid_projections, 1] * height_shift_mult) + query_projection_ids[v, valid_projections, 0]

            # Calculate depth image values at valid locations and filter outliers
            depth_diff_at_projections = torch.abs(flat_depth[query_projection_flat_inds] - image_space_z[v, valid_projections])
            valid_depth_values = depth_diff_at_projections < depth_treshold

            # Copy valid features from image to queries
            projected_point_features[sample_indexer[batch_mask][valid_projections][valid_depth_values], v, :] = flat_features[query_projection_flat_inds[valid_depth_values], :]

    return projected_point_features

def assign_image_features_to_point_cloud_full_scenes(batched_cloud, image_features, camera_poses, camera_intrinsics, depth_images, depth_treshold=2.5, projected_point_features=None, sample_indexer=None):
    # reference equation at https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

    batch_size = int(batched_cloud[-1, 0]) + 1
    sampled_point_num = batched_cloud.shape[0]
    feature_dim = image_features.shape[-1]
    view_num = image_features.shape[1]
    height = image_features.shape[2]
    width = image_features.shape[3]
    device = batched_cloud.device

    # Coordinate frame transform from pose to camera coords
    S_img = torch.Tensor([[-1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0]]).to(device)

    # we need these to properly index the image feature tensor
    batch_shift_mult = view_num * height * width
    view_shift_mult = height * width
    height_shift_mult = width
    flat_features = image_features.view(-1, feature_dim)
    flat_depth = depth_images.flatten()

    # To store values
    if projected_point_features is None:
        projected_point_features = torch.zeros((sampled_point_num, view_num, feature_dim), dtype=flat_features.dtype, device=flat_features.device)
    else:
        projected_point_features = projected_point_features * 0.

    # Iterate over all batches
    for batch_num in range(batch_size):
        intrinsics = camera_intrinsics[batch_num].flatten()
        homo_coords = batched_cloud.clone()
        homo_coords[:, :3] = homo_coords[:, 1:]
        homo_coords[:, 3] = 1.
        K_img = torch.Tensor([[intrinsics[0], 0, intrinsics[2]],
                              [0, intrinsics[1], intrinsics[3]],
                              [0, 0, 1]]).to(device)

        batched_intrinsics = (K_img @ S_img).unsqueeze(0).expand(view_num, -1, -1)
        query_projection_ids = (torch.bmm(batched_intrinsics, camera_poses[batch_num].inverse()) @ homo_coords.T).permute(0, 2, 1)
        image_space_z = - query_projection_ids[:, :, 2]
        query_projection_ids = torch.round((query_projection_ids[:, :, :2] / query_projection_ids[:, :, 2].unsqueeze(-1))).long()
        query_projection_ids[:, :, 1] = height - query_projection_ids[:, :, 1]  # invert y indexing

        # iterate over all views to get valid ids with projections within the frustrum
        for v in range(view_num):
            valid_width = torch.logical_and(query_projection_ids[v, :, 0] >= 0, query_projection_ids[v, :, 0] < width)
            valid_height = torch.logical_and(query_projection_ids[v, :, 1] >= 0, query_projection_ids[v, :, 1] < height)
            valid_projections = torch.logical_and(valid_height, valid_width)

            # Calculate shaped coordinates to flat coordinates
            query_projection_flat_inds = (batch_num * batch_shift_mult) + (v * view_shift_mult) + (query_projection_ids[v, valid_projections, 1] * height_shift_mult) + query_projection_ids[v, valid_projections, 0]

            # Calculate depth image values at valid locations and filter outliers
            depth_diff_at_projections = torch.abs(flat_depth[query_projection_flat_inds] - image_space_z[v, valid_projections])
            valid_depth_values = depth_diff_at_projections < depth_treshold

            # Copy valid features from image to queries
            projected_point_features[sample_indexer[valid_projections][valid_depth_values], v, :] = flat_features[query_projection_flat_inds[valid_depth_values], :]

    return projected_point_features


def feature_sim(output_feats, anchor_feats, config):

    # Calculate cosine similarity
    An = F.normalize(output_feats, p=2, dim=1)
    Bn = F.normalize(anchor_feats, p=2, dim=1).unsqueeze(0).expand(An.shape[0], anchor_feats.shape[0],
                                                                   anchor_feats.shape[1])
    Dcos = torch.bmm(An.unsqueeze(1), Bn.transpose(1, 2))
    return Dcos.squeeze()


def sampled_preds_to_voxels(sampled_coords, sampled_preds, sampled_targets, voxel_coords, config, assign_mode='knn', projected_points=None):
    from torch_cluster import nearest

    voxel_preds = torch.zeros(voxel_coords.shape[0], dtype=torch.long, device=voxel_coords.device)
    voxel_targets = torch.zeros(voxel_coords.shape[0], dtype=torch.long, device=voxel_coords.device) + config.data.ignore_label
    projected_voxels = torch.ones(voxel_coords.shape[0], dtype=torch.bool, device=voxel_coords.device)

    if assign_mode == 'knn':

        # iterate over batches
        target = sampled_coords[:, 1:]
        target_batch = sampled_coords[:, 0].long()

        clustered_voxels = nearest(voxel_coords[:, 1:].float(), target, voxel_coords[:, 0].long(), target_batch)
        voxel_preds = sampled_preds[clustered_voxels]
        voxel_targets = sampled_targets[clustered_voxels]
        if projected_points is not None:
            projected_voxels = projected_points[clustered_voxels]

    elif assign_mode == 'majority':
        clustered_voxels = None

    return voxel_preds, voxel_targets, clustered_voxels


def visualize_features(points, feats_3d, feats_2d, feats_pos_enc, local_variations=True):

    import open3d as o3d
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)  # for rgb

    projected_feats_3d = pca.fit_transform(feats_3d.cpu().numpy())
    projected_feats_2d = pca.fit_transform(feats_2d.cpu().numpy())
    projected_feats_pos_enc = pca.fit_transform(feats_pos_enc.cpu().numpy())

    # normalize between 0, 1
    projected_feats_3d = projected_feats_3d - projected_feats_3d.min()
    projected_feats_3d = projected_feats_3d / projected_feats_3d.max()

    projected_feats_2d = projected_feats_2d - projected_feats_2d.min()
    projected_feats_2d = projected_feats_2d / projected_feats_2d.max()

    projected_feats_pos_enc = projected_feats_pos_enc - projected_feats_pos_enc.min()
    projected_feats_pos_enc = projected_feats_pos_enc / projected_feats_pos_enc.max()

    # visualize open3d
    shift = (points[:, 1].max() - points[:, 1].min()).item() * 1.2
    pcd_3d = o3d.geometry.PointCloud()
    pcd_3d.points = o3d.utility.Vector3dVector(points[:, 1:].cpu().numpy() + [shift, 0., 0.])
    pcd_3d.colors = o3d.utility.Vector3dVector(projected_feats_3d)

    pcd_2d = o3d.geometry.PointCloud()
    pcd_2d.points = o3d.utility.Vector3dVector(points[:, 1:].cpu().numpy())
    pcd_2d.colors = o3d.utility.Vector3dVector(projected_feats_2d)

    pcd_pos = o3d.geometry.PointCloud()
    pcd_pos.points = o3d.utility.Vector3dVector(points[:, 1:].cpu().numpy() - [shift, 0., 0.])
    pcd_pos.colors = o3d.utility.Vector3dVector(projected_feats_pos_enc)

    vis_pcd = [pcd_3d, pcd_2d, pcd_pos]

    # we also want to extract the local variations of the different features
    if local_variations:
        point_num, feature_dim = feats_3d.shape
        NN_num = 8
        cut_off_tresh = 0.1

        similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        tree = KDTree(points.cpu().numpy(), leafsize=8)
        NNs = tree.query(points.cpu().numpy(), k=NN_num + 1)[1][:, 1:]  # we are interested in the 8 neighbours

        NN_features = feats_3d[NNs].view(-1, feature_dim)  # should be [point_num * NN_num, feature_dim]
        NN_similarities = (NN_features - feats_3d.repeat_interleave(NN_num, dim=0)).sum(-1).view(point_num, NN_num)

        NN_std = NN_similarities.std(dim=1)
        # NN_std[NN_std > cut_off_tresh] = cut_off_tresh
        NN_std = torch.log(NN_std + 10e-4)
        NN_std = NN_std - NN_std.min()
        NN_std = (NN_std / NN_std.max()).cpu().numpy()
        NN_std_colored = np.stack((NN_std, 1. - NN_std, np.zeros(point_num))).T

        # variation cloud
        variation_pcd = o3d.geometry.PointCloud()
        variation_pcd.points = o3d.utility.Vector3dVector(points[:, 1:].cpu().numpy() + [2 * shift, 0., 0.])
        variation_pcd.colors = o3d.utility.Vector3dVector(NN_std_colored)
        vis_pcd += [variation_pcd]

    o3d.visualization.draw_geometries(([*vis_pcd]))

def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
    """Get permutation from pointcloud to input voxel coords."""

    def _hash_coords(coords, coords_min, coords_dim):
        return np.ravel_multi_index((coords - coords_min).T, coords_dim)

    # Validate input.
    input_batch_size = input_coords[:, -1].max().item()
    pointcloud_batch_size = pointcloud[:, -1].max().int().item()
    transformation_batch_size = transformation[:, -1].max().int().item()
    assert input_batch_size == pointcloud_batch_size == transformation_batch_size
    pointcloud_permutation, pointcloud_target = [], []

    # Process each batch.
    for i in range(input_batch_size + 1):
        # Filter batch from the data.
        input_coords_mask_b = input_coords[:, -1] == i
        input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
        pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
        transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
        # Transform original pointcloud to voxel space.
        original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
        original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
        # Hash input and voxel coordinates to flat coordinate.
        vcoords_all = np.vstack((input_coords_b, original_vcoords))
        vcoords_min = vcoords_all.min(0)
        vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
        input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
        original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
        # Query voxel predictions from original pointcloud.
        key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
        pointcloud_permutation.append(
            np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
        pointcloud_target.append(pointcloud_b[:, -1].astype(int))
    pointcloud_permutation = np.concatenate(pointcloud_permutation)
    # Prepare pointcloud permutation array.
    pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
    permutation_mask = pointcloud_permutation >= 0
    permutation_valid = pointcloud_permutation[permutation_mask]
    # Permute voxel output to pointcloud.
    pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
    pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
    # Permute voxel prediction to pointcloud.
    # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
    pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
    pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
    # Map pointcloud target to respect dataset IGNORE_LABELS
    pointcloud_target = torch.from_numpy(
        np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
    return pointcloud_output, pointcloud_pred, pointcloud_target


def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian', eps=10e-9, nms_thr=0.5):


    if kernel == 'mask':
        n_samples = len(cate_scores)
        if n_samples == 0:
            return []

        keep = seg_masks.new_ones(cate_scores.shape)
        seg_masks = seg_masks.float()

        for i in range(n_samples - 1):
            if not keep[i]:
                continue
            mask_i = seg_masks[i]
            label_i = cate_labels[i]
            for j in range(i + 1, n_samples, 1):
                if not keep[j]:
                    continue
                mask_j = seg_masks[j]
                label_j = cate_labels[j]
                if label_i != label_j:
                    continue
                # overlaps
                inter = (mask_i * mask_j).sum()
                union = sum_masks[i] + sum_masks[j] - inter
                if union > 0:
                    if inter / union > nms_thr:
                        keep[j] = False
                else:
                    keep[j] = False

        cate_scores[~keep] = 0.0
        return cate_scores

    else:
        n_samples = len(cate_labels)
        if n_samples == 0:
            return []
        if False:
            seg_masks = seg_masks.clone()
            ori_masks = seg_masks.clone()
            seg_masks[ori_masks.max(1, keepdim=True)[0].expand(-1, seg_masks.shape[1], -1)] = True
            ori_masks[ori_masks.max(2, keepdim=True)[0].expand(-1, -1, seg_masks.shape[2])] = True
            seg_masks *= ori_masks

        # inter.
        seg_masks = seg_masks.float()
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix + eps)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

        # IoU decay / soft nms
        delay_iou = iou_matrix * label_matrix

        # matrix nms
        if kernel == 'linear':
            delay_matrix = (1 - delay_iou) / (1 - compensate_iou + eps)
            delay_coefficient, _ = delay_matrix.min(0)
        else:
            delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
            compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
            delay_coefficient, _ = (delay_matrix / (compensate_matrix + eps)).min(0)

        # update the score.q
        cate_scores_update = cate_scores * delay_coefficient

        return cate_scores_update
