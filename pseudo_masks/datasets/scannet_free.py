import numpy as np
import os.path

from lib.datasets.scannet_solo import *


class ScanNetFree_Dataset(ScanNetSOLO_Dataset):

    CLASS_LABELS = ['background', 'foreground']
    VALID_CLASS_IDS = [0, 1]
    INSTANCE_IGNORE_LABELS = [0]
    ORACLE_INSTANCE_IGNORE_LABELS = [-1, 0, 1, 2]
    NUM_LABELS = 2
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 target_transform=None,
                 input_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 phase='train',
                 cache=False,
                 data_root=None):

        # Init original ScanNet_SOLO dataset
        super().__init__(
            config=config,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            target_transform=target_transform,
            input_transform=input_transform,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            phase=phase,
            cache=cache)

        # Create label map
        label_map = {}
        for index in range(2):  # we only have FG and BG
            label_map[index] = index
        self.label_map = label_map
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])
        self.NUM_LABELS = 2

        self.mapped_instance_ignore_labels = self.INSTANCE_IGNORE_LABELS

        self.CLASS_LABELS_INSTANCE = ['foreground']
        self.VALID_CLASS_IDS_INSTANCE = [1]
        self.IGNORE_LABELS_INSTANCE = sorted(list(set.union(set(self.IGNORE_LABELS), set(self.INSTANCE_IGNORE_LABELS))))

        # Add Oracle Label map too
        self.use_oracle_mode = True if (isinstance(self.config.freemask.oracle_mode, bool) and self.config.freemask.oracle_mode) or 'val' in self.phase else False
        self.label_map_oracle = {}
        for index, row in self.labels_pd.iterrows():
            id = row['id']
            nyu40id = row['nyu40id']
            if nyu40id in VALID_CLASS_IDS_20:
                self.label_map_oracle[id] = nyu40id
            else:
                self.label_map_oracle[id] = self.ignore_mask

        # Add ignore
        self.label_map_oracle[0] = self.ignore_mask
        self.label_map_oracle[self.ignore_mask] = self.ignore_mask
        self.oracle_label_mapper = np.vectorize(lambda x: self.label_map_oracle[x])

    def load_unsupervised_scene_data(self, index):

        # Load masks and clouds from numpy files
        filepath = self.data_root / self.data_paths[index]
        scene_name = filepath.stem

        try:
            pointcloud = np.load(f'{filepath}_cloud.npy')
        except:
            print(f'Could not load scene cloud from {filepath}_cloud.npy')
            return self.load_unsupervised_scene_data(np.random.randint(len(self.data_paths)))

        coords = pointcloud[:, :3].astype(np.float32)
        feats = (pointcloud[:, 3:6]).astype(int)

        # Load intrinsics info, but shouldn't align cloud as already done
        # coords, axis_alignment, info_dict = self.load_intrinsics(scene_name, coords)
        info_file = os.path.join(self.config.data.scannet_images_path, f'{scene_name}', f'{scene_name}.txt')
        info_dict = {}
        with open(info_file) as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for line in f:
                    (key, val) = line.split(" = ")
                    info_dict[key] = np.fromstring(val, sep=' ')
        axis_alignment = info_dict['axisAlignment'].reshape(4, 4) if ('axisAlignment' in info_dict and self.config.data.align_scenes) else np.identity(4)

        if self.use_oracle_mode:
            # In oracle experiments we load the original ScanNet instances and treat them as GT labels
            gt_scene_data = self.load_scene_data(index, self.config.freemask.oracle_data_root)
            coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity = gt_scene_data

            oracle_labels = self.oracle_label_mapper(labels)
            valid_inst_mask = ~np.isin(oracle_labels, self.ORACLE_INSTANCE_IGNORE_LABELS) * (instance_ids >= 0)
            unique_instances = np.unique(instance_ids[valid_inst_mask])

            hard_masks = np.zeros((coords.shape[0], len(unique_instances))).astype(bool)
            for inst_id, inst_idx in enumerate(unique_instances):
                hard_masks[:, inst_id] = instance_ids == inst_idx

        else:  # true freemask mode
            try:
                # Load and discretize soft masks with threshold value
                freemasks = np.load(f'{filepath}_masks.npy')
                hard_masks = (freemasks >= self.config.freemask.hard_mask_threshold).astype(bool)
            except:
                # If no masks, return random scene
                print(f'Could not load freemasks from {filepath}_masks.npy')
                return self.load_unsupervised_scene_data(np.random.randint(len(self.data_paths)))

        # Filter masks by ratio if instance to scene extents
        filtered_masks = []
        scene_extents = coords.max(0) - coords.min(0)
        for mask_id in range(hard_masks.shape[1]):
            if np.all(~hard_masks[:, mask_id]):
                continue
            inst_coords = coords[hard_masks[:, mask_id]]
            inst_extent = inst_coords.max(0) - inst_coords.min(0)

            # Skip if extent is too large
            extent_ratios = inst_extent / scene_extents
            if np.any(extent_ratios[:2] > self.config.freemask.instance_to_scene_max_ratio):  # we only care about XY extent
                continue
            else:
                filtered_masks += [mask_id]
        hard_masks = hard_masks[:, filtered_masks]

        # Labels should be foreground where masks are present
        labels = np.zeros(coords.shape[0]).astype(int)
        labels[hard_masks.sum(-1) != 0] = 1

        # Parse freemasks instance ids everywhere where mask was true
        instance_ids = np.zeros(coords.shape[0]).astype(int)
        for mask_id in range(hard_masks.shape[1]):
            instance_ids[hard_masks[:, mask_id]] = mask_id + 1

        # Load RGB data and segment mesh if requested
        rgb_images, poses, color_intrinsics = self.load_rgb_data(scene_name, info_dict, axis_alignment)
        segment_ids, seg_connectivity = self.segment_mesh(scene_name, coords, axis_alignment)

        return coords, feats, labels, instance_ids, scene_name, rgb_images, poses, color_intrinsics, segment_ids, seg_connectivity

    def __getitem__(self, index):

        # Load unsupervised data
        coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity = self.load_unsupervised_scene_data(index)

        # Voxelize and augment
        coords, feats, labels, instance_ids, camera_poses, segment_ids, transformations = self.prepare_scene_data(coords, feats, labels, instance_ids, camera_poses, segment_ids)

        # Generate grid targets for SOLO-style
        cloud_instances, grid_locations_dict, instance_ids, grid_indices = self.generate_grid_targets(coords, labels, instance_ids, segment_ids)

        # Collect in tuple
        return_args = (coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations[1].astype(np.float32))
        extended_return_args = return_args + (cloud_instances, grid_locations_dict, grid_indices)

        return extended_return_args

class ScanNetFree_2cmDataset(ScanNetFree_Dataset):
    VOXEL_SIZE = 0.02
