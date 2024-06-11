import glob
import logging
import os
import warnings
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
import MinkowskiEngine as ME
from PIL import Image
import open3d as o3d

from datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from utils.pc_utils import read_plyfile, save_point_cloud, load_labels
from utils.utils import read_txt, fast_hist, per_class_iu, load_matrix_from_txt

from constants.scannet_constants import *
from constants.dataset_sets import *
from scipy.spatial import KDTree

import felzenszwalb_cpp

class ScanNet_Dataset(VoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_20
    CLASS_LABELS = CLASS_LABELS_20
    VALID_CLASS_IDS = VALID_CLASS_IDS_20
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 verbose=False,
                 phase: str = 'train',
                 data_root: str = None,
                 **kwargs):

        if 'train' not in phase.lower():
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        self.config = config
        self.phase = phase

        data_root = config.data.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        self.data_paths = np.array(sorted(read_txt(os.path.join(data_root, f'{phase}.txt'))))
        self.scene_names = [Path(data_path).stem for data_path in self.data_paths]
        logging.info('Loading {} scenes into {} for {} phase'.format(len(self.data_paths), self.__class__.__name__, phase))
        self.chunked_data = 'chunk' in self.config.data.scannet_path

        # Init DatasetBase
        super().__init__(
            self.data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.data.ignore_label,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config,
            cache=cache)

        # Load dataframe with label map
        labels_pd = pd.read_csv(os.path.join('constants', 'scannetv2-labels.combined.tsv'), sep='\t', header=0)
        self.labels_pd = labels_pd

        # Create label map
        label_map = {}
        for index, row in labels_pd.iterrows():
            id = row['id']
            nyu40id = row['nyu40id']
            if nyu40id in self.VALID_CLASS_IDS:
                scannet20_index = self.VALID_CLASS_IDS.index(nyu40id)
                label_map[id] = scannet20_index
            else:
                label_map[id] = self.ignore_mask

        # Add ignore
        label_map[0] = self.ignore_mask
        label_map[self.ignore_mask] = self.ignore_mask
        self.label_map = label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])
        self.NUM_LABELS = len(self.VALID_CLASS_IDS)

        # Precompute a mapping from ids to categories
        self.id2cat_name = {}
        self.cat_name2id = {}
        for id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            self.id2cat_name[id] = cat_name
            self.cat_name2id[cat_name] = id

        # Add image data if requested
        if config.image_data.use_images:

            # 2D image shape from config
            scale = self.config.image_data.downsample_ratio
            self.depth_shape = tuple(int(scale * dim) for dim in self.config.image_data.image_resolution)
            self.pil_depth_shape = tuple(reversed(self.depth_shape))

            self.image_transform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                                       transforms.Resize([self.depth_shape[0], self.depth_shape[1]])])

    def load_image(self, path, target_shape, label=False):
        if label:
            image = Image.open(path).resize(target_shape, Image.NEAREST)
        else:
            image = Image.open(path).convert('RGB').resize(target_shape, Image.BILINEAR)

        return np.array(image).astype(int)

    def load_rgb_data(self, scene_name, info_dict, axis_alignment):

        rgb_images, poses, color_intrinsics, segment_ids, seg_connectivity = None, None, None, None, None

        # Load image data too if requested
        if self.config.image_data.use_images:
            # Load intrinsics
            orig_color_shape = np.array([info_dict['colorHeight'], info_dict['colorWidth']])
            color_scale = [self.depth_shape[0] / orig_color_shape[0], self.depth_shape[1] / orig_color_shape[1]]
            color_intrinsics = np.array([info_dict['fx_color'] * color_scale[0],
                                         info_dict['fy_color'] * color_scale[1],
                                         info_dict['mx_color'] * color_scale[0],
                                         info_dict['my_color'] * color_scale[1]])

            # Load rgb images
            RGB_PATH = self.config.data.scannet_images_path + f'/{scene_name}/color/'
            rgb_paths = sorted(glob.glob(RGB_PATH + '*.jpg'))
            rgb_images = np.array([self.load_image(rgb_path, self.pil_depth_shape) for rgb_path in rgb_paths])
            rgb_images = torch.from_numpy(rgb_images / 255.).permute(0, 3, 1, 2).float()
            rgb_images = self.image_transform(rgb_images)

            # Load camera poses
            POSE_PATH = self.config.data.scannet_images_path + f'/{scene_name}/pose/'
            pose_paths = sorted(glob.glob(POSE_PATH + '*.txt'))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                poses = np.array([axis_alignment @ load_matrix_from_txt(pose_path) for pose_path in pose_paths])

        return rgb_images, poses, color_intrinsics

    def segment_mesh(self, scene_name, coords, axis_alignment, scene_mesh=None):
        segment_ids, all_seg_connectivity, all_seg_indices = None, None, None

        if self.config.data.segments_as_grids:

            if scene_mesh is None:
                # Load mesh and segments file to load segment
                mesh_file = os.path.join(self.config.data.scannet_images_path, f'{scene_name}', f'{scene_name}_vh_clean_2.ply')
                scene_mesh = o3d.io.read_triangle_mesh(mesh_file)
                scene_mesh = scene_mesh.transform(axis_alignment)

                if self.chunked_data:  # crop mesh to make it faster
                    # Get scene dimensions and crop mesh to size
                    scene_mins, scene_maxes = np.floor(coords.min(0)), np.floor(coords.max(0)) + 1
                    chunk_crop = o3d.geometry.AxisAlignedBoundingBox(scene_mins, scene_maxes)
                    scene_mesh = scene_mesh.crop(chunk_crop)

            # Parse to correct type and call algorithm
            vertices = np.array(scene_mesh.vertices).astype(np.single)
            colors = np.array(scene_mesh.vertex_colors).astype(np.single)
            faces = np.array(scene_mesh.triangles).astype(np.intc)

            all_seg_indices, all_seg_connectivity = [], []
            seg_min_point_nums = self.config.data.segments_min_vert_nums
            for seg_min_point_num in seg_min_point_nums:

                seg_indices, seg_connectivity = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, 0.005, seg_min_point_num)

                # Find mapping between voxels and segment ids and save to return
                # this is a heavy part sadly, but I don't know any other idea
                vertices = np.array(scene_mesh.vertices)
                mesh_tree = KDTree(vertices)
                _, mesh_voxel_matches = mesh_tree.query(coords, k=1)
                mesh_voxel_matches = mesh_voxel_matches.flatten()

                segment_ids = seg_indices[mesh_voxel_matches]
                all_seg_indices += [segment_ids]
                all_seg_connectivity += [seg_connectivity]

            all_seg_indices = np.stack(all_seg_indices, axis=-1)

        return all_seg_indices, all_seg_connectivity

    def load_intrinsics(self, scene_name, coords):

        # Load intrinsics info
        info_file = os.path.join(self.config.data.scannet_images_path, f'{scene_name}', f'{scene_name}.txt')
        info_dict = {}
        with open(info_file) as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for line in f:
                    (key, val) = line.split(" = ")
                    info_dict[key] = np.fromstring(val, sep=' ')
        axis_alignment = info_dict['axisAlignment'].reshape(4, 4) if ('axisAlignment' in info_dict and self.config.data.align_scenes) else np.identity(4)

        # Also rotate the orig scene with it
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords = homo_coords @ axis_alignment.T[:, :3]

        return coords, axis_alignment, info_dict

    def load_scene_data(self, index, data_root=None):

        coords, feats, labels, instance_ids, scene_name = self.load_data(index, data_root=data_root)

        if self.chunked_data:  # crop mesh to make it faster
            scene_name = scene_name[:scene_name.rfind("_")]

        # Load intrinsics info and align cloud
        coords, axis_alignment, info_dict = self.load_intrinsics(scene_name, coords)

        # Load RGB data and segment mesh if requested
        rgb_images, poses, color_intrinsics = self.load_rgb_data(scene_name, info_dict, axis_alignment)
        segment_ids, seg_connectivity = self.segment_mesh(scene_name, coords, axis_alignment)

        return coords, feats, labels, instance_ids, scene_name, rgb_images, poses, color_intrinsics, segment_ids, seg_connectivity

    def prepare_scene_data(self, coords, feats, labels, instance_ids, camera_poses, segment_ids):

        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            _, inds = ME.utils.sparse_quantize(coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            coords = coords[inds]
            feats = feats[inds]
            labels = labels[inds]
            instance_ids = instance_ids[inds]
            segment_ids = segment_ids[inds] if segment_ids is not None else None

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, _ = self.prevoxel_transform(coords, feats, None)  # labels and instances are unchanged for these transforms

        # Voxelize as usual
        coords, feats, voxelized_inds, transformations = self.voxelizer.voxelize(coords, feats)
        labels = labels[voxelized_inds]
        instance_ids = instance_ids[voxelized_inds] if instance_ids is not None else None
        segment_ids = segment_ids[voxelized_inds] if segment_ids is not None else None

        if camera_poses is not None:
            scale_transform, rotation_transform = transformations
            camera_poses[:, :, 3] = camera_poses[:, :, 3] @ (scale_transform @ rotation_transform).T
            camera_poses[:, :3, :3] = rotation_transform[:3, :3] @ camera_poses[:, :3, :3]

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, dropout_map = self.input_transform(coords, feats, np.arange(coords.shape[0]))
            labels = labels[dropout_map]
            instance_ids = instance_ids[dropout_map] if instance_ids is not None else None
            segment_ids = segment_ids[dropout_map] if segment_ids is not None else None

            rand_shift = (np.random.rand(3) * 100).astype(coords.dtype)
            coords += rand_shift
            if camera_poses is not None:
                camera_poses[:, :3, 3] += rand_shift

        if self.target_transform is not None:
            coords, feats, dropout_map = self.target_transform(coords, feats, np.arange(coords.shape[0]))
            labels = labels[dropout_map]
            instance_ids = instance_ids[dropout_map] if instance_ids is not None else None
            segment_ids = segment_ids[dropout_map] if segment_ids is not None else None

        if self.IGNORE_LABELS is not None:
            labels = self.label_mapper(labels)

        feats = feats / 255. - 0.5  # normalize_color

        return coords, feats, labels, instance_ids, camera_poses, segment_ids, transformations

    def __getitem__(self, index):

        # Load data
        coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity = self.load_scene_data(index)

        # Voxelize and augment
        coords, feats, labels, instance_ids, camera_poses, segment_ids, transformations = self.prepare_scene_data(coords, feats, labels, instance_ids, camera_poses, segment_ids)

        # Collect in tuple
        return_args = (coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations[1].astype(np.float32))

        return return_args


    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def get_classids(self):
        return self.VALID_CLASS_IDS

    def get_classnames(self):
        return self.CLASS_LABELS

    def _augment_locfeat(self, pointcloud):
        # Assuming that pointcloud is xyzrgb(...), append location feat.
        pointcloud = np.hstack(
            (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
             pointcloud[:, 6:]))
        return pointcloud

    def scene_name_to_index(self, scene_name):
        return self.scene_names.index(scene_name)

    def index_to_scene_name(self, index):
        return self.scene_names[index]

    def get_full_cloud_by_scene_name(self, scene_name):

        scene_data = self.load_scene_data(self.scene_name_to_index(scene_name))
        return scene_data


class ScanNet_2cmDataset(ScanNet_Dataset):
    VOXEL_SIZE = 0.02


class ScanNet_10cmDataset(ScanNet_Dataset):
    VOXEL_SIZE = 0.10

