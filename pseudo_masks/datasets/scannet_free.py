import numpy as np
import os.path
import warnings

from datasets.scannet import *
from utils.utils import OrderedSet

class CloudInstance(object):

    def __init__(self, instance_id, instance_label, coords, cloud_instances, class_label=-1):

        self.class_label = class_label
        self.instance_id = instance_id
        self.instance_label = instance_label

        self.inst_mask = cloud_instances == instance_label
        self.num_points = np.sum(self.inst_mask)
        self.instance_points = coords[self.inst_mask]
        self.bb = [np.min(self.instance_points, axis=0), np.max(self.instance_points, axis=0)]
        self.center_of_mass = np.mean(self.instance_points, axis=0)
        self.grid_locations = []

    def add_grid_locations(self, location):
        self.grid_locations.append(location)


class ScanNetFree_Dataset(ScanNet_Dataset):

    CLASS_LABELS = ['background', 'foreground']
    VALID_CLASS_IDS = [0, 1]
    INSTANCE_IGNORE_LABELS = [0]
    ORACLE_INSTANCE_IGNORE_LABELS = [-1, 0, 1, 2]
    NUM_LABELS = 2
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    INSTANCE_IGNORE_LABELS = (1, 2)  # wall and floor for scannet20

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

        # Init original ScanNet dataset
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

        # Define the grid sizes
        grid_sizes_in_m = np.array(self.config.SOLO3D.grid_sizes)
        self.output_voxel_resolutions = np.array([1, 2, 4, 8])
        output_voxel_sizes = self.output_voxel_resolutions * self.VOXEL_SIZE
        self.grid_sizes_in_voxel = grid_sizes_in_m / output_voxel_sizes[0]

        self.mapped_instance_ignore_labels = [self.VALID_CLASS_IDS.index(scannet20_inst_id) for scannet20_inst_id in self.INSTANCE_IGNORE_LABELS]

        self.CLASS_LABELS_INSTANCE = list(OrderedSet(self.CLASS_LABELS) - OrderedSet(PARENT_CLASS_SUPERCAT))
        self.VALID_CLASS_IDS_INSTANCE = list(OrderedSet(self.VALID_CLASS_IDS) - OrderedSet(self.INSTANCE_IGNORE_LABELS))
        self.IGNORE_LABELS_INSTANCE = sorted(list(set.union(set(self.IGNORE_LABELS), set(self.INSTANCE_IGNORE_LABELS))))

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

    def generate_grid_targets(self, coords, labels, instance_ids, segment_ids=None):

        # First generate the instances with corresponding information, see in CloudInstance
        cloud_instances = []
        mapped_instance_ids = np.ones_like(instance_ids) * self.config.data.ignore_label
        inst_label_to_id = {}
        inst_id = 1
        for unique_inst_id in np.unique(instance_ids):

            # get class label base don uniform instance mask
            class_label = labels[instance_ids == unique_inst_id][0]

            if class_label in self.mapped_instance_ignore_labels or class_label == self.config.data.ignore_label:  # we don't care if it is wall or floor or ignore label
                continue
            else:
                cloud_instance = CloudInstance(inst_id, unique_inst_id, coords, instance_ids, class_label)
                mapped_instance_ids[cloud_instance.inst_mask] = inst_id
                cloud_instances += [cloud_instance]
                inst_label_to_id[unique_inst_id] = inst_id - 1  # starts from 1, but used for indexing
                inst_id += 1

        # Generate all the grids and corresponding target information for supervision
        multi_res_grid_locations_dict = []
        grid_indices = np.ones((coords.shape[0], len(self.grid_sizes_in_voxel)), dtype=int) * self.config.data.ignore_label

        # Iterate over all resolution to get grid locations at different grid sizes
        for res_id, grid_size_in_voxel in enumerate(self.grid_sizes_in_voxel):
            grid_locations_dict = {}

            if not self.config.SOLO3D.segments_as_grids:  # we use standard grids as a sliding window

                # Get scene dimensions
                scene_mins, scene_maxes = np.floor(coords.min(0)), np.floor(coords.max(0))

                x_low, y_low, z_low = scene_mins
                x_max, y_max, z_max = scene_maxes + 1
                locations_id_x, location_id_y, location_id_z, grid_index = 0, 0, 0, 0
                while x_low < (x_max - grid_size_in_voxel) or x_low == scene_mins[0]:
                    points_in_grid_x = (coords[:, 0] >= x_low) * (coords[:, 0] < (x_low + grid_size_in_voxel))
                    while y_low < (y_max - grid_size_in_voxel) or y_low == scene_mins[1]:
                        points_in_grid_y = points_in_grid_x * (coords[:, 1] >= y_low) * (coords[:, 1] < (y_low + grid_size_in_voxel))
                        while z_low < (z_max - grid_size_in_voxel) or z_low == scene_mins[2]:

                            # check if at least min number of points within the grid cell
                            points_in_grid = points_in_grid_y * (coords[:, 2] >= z_low) * (coords[:, 2] < (z_low + grid_size_in_voxel))  # type: np.ndarray

                            if points_in_grid.sum() >= self.config.SOLO3D.grid_min_point_num:

                                # Update the grid indices for easier slicing for grid points
                                grid_indices[points_in_grid, res_id] = grid_index

                                # for every instance add the valid grid location which covers the part of the instance
                                # For every grid location add the instance mask
                                unique_valid_inst_labels_in_grid = set.intersection(set(np.unique(instance_ids[points_in_grid])), set(inst_label_to_id.keys()))
                                max_inst_point_num_in_grid = 0
                                for inst_in_grid in unique_valid_inst_labels_in_grid:
                                    if inst_in_grid in inst_label_to_id:  # this means the instance is a valid instance
                                        cloud_inst = cloud_instances[inst_label_to_id[inst_in_grid]]  # type: CloudInstance
                                        cloud_inst.add_grid_locations((locations_id_x, location_id_y, location_id_z))
                                        cloud_instances[inst_label_to_id[inst_in_grid]] = cloud_inst

                                        # Add the grid category and mask if the instance is the majority in the grid
                                        grid_num_inst_points = (points_in_grid * (instance_ids == inst_in_grid)).sum()
                                        if grid_num_inst_points > max_inst_point_num_in_grid:
                                            grid_target = {
                                                'bb': [x_low, y_low, z_low, x_low + grid_size_in_voxel, y_low + grid_size_in_voxel,
                                                       z_low + grid_size_in_voxel],
                                                'label': cloud_inst.class_label,
                                                'mask': cloud_inst.inst_mask,
                                                'inst_id': cloud_inst.instance_id,
                                                'grid_index': grid_index}
                                            grid_locations_dict[(locations_id_x, location_id_y, location_id_z)] = grid_target
                                            max_inst_point_num_in_grid = grid_num_inst_points

                                # If no instance was covered the box is still valid, but label should be the panoptic category
                                if max_inst_point_num_in_grid == 0 and points_in_grid.any():
                                    label_nums = np.bincount(labels[points_in_grid * (labels != self.config.data.ignore_label)])
                                    if len(label_nums) > 0:
                                        majority_label = label_nums.argmax()
                                        grid_target = {'bb': [x_low, y_low, z_low, x_low + grid_size_in_voxel, y_low + grid_size_in_voxel, z_low + grid_size_in_voxel],
                                                       'label': majority_label,
                                                       'mask': np.zeros(coords.shape[0], dtype=bool),
                                                       'grid_index': grid_index}
                                        grid_locations_dict[(locations_id_x, location_id_y, location_id_z)] = grid_target

                                # Update the grid index only for valid locations
                                grid_index += 1

                            # last grid should be overlapping
                            z_low += grid_size_in_voxel
                            if z_low >= z_max:
                                z_low = z_max - grid_size_in_voxel + 10e-6
                            location_id_z += 1


                        y_low += grid_size_in_voxel
                        if y_low >= y_max:
                            y_low = y_max - grid_size_in_voxel + 10e-6
                        location_id_y += 1
                        z_low, location_id_z = scene_mins[2], 0

                    x_low += grid_size_in_voxel
                    locations_id_x += 1
                    y_low, location_id_y = scene_mins[1], 0

                    # last grid should be overlapping
                    if x_low >= (x_max - grid_size_in_voxel):
                        x_low = x_max - grid_size_in_voxel + 10e-6
            else:  # use the segments as grid proposals
                segment_ids_at_res = segment_ids[:, res_id]
                unique_segments = np.unique(segment_ids_at_res)
                for segment_id in unique_segments:

                    points_in_segment = segment_ids_at_res == segment_id

                    if points_in_segment.sum() >= self.config.SOLO3D.grid_min_point_num:

                        # Update the grid indices for easier slicing for grid points
                        grid_indices[points_in_segment, res_id] = segment_id

                        # for every instance add the valid grid location which covers the part of the instance
                        # For every grid location add the instance mask
                        unique_valid_inst_labels_in_grid = set.intersection(set(np.unique(instance_ids[points_in_segment])), set(inst_label_to_id.keys()))
                        max_inst_point_num_in_grid = 0
                        for inst_in_grid in unique_valid_inst_labels_in_grid:
                            if inst_in_grid in inst_label_to_id:  # this means the instance is a valid instance
                                cloud_inst = cloud_instances[inst_label_to_id[inst_in_grid]]  # type: CloudInstance
                                cloud_inst.add_grid_locations(segment_id)
                                cloud_instances[inst_label_to_id[inst_in_grid]] = cloud_inst

                                # Add the grid category and mask if the instance is the majority in the grid
                                grid_num_inst_points = (points_in_segment * (instance_ids == inst_in_grid)).sum()
                                if grid_num_inst_points > max_inst_point_num_in_grid:
                                    grid_target = {'label': cloud_inst.class_label,
                                                   'mask': cloud_inst.inst_mask,
                                                   'inst_id': cloud_inst.instance_id,
                                                   'grid_index': segment_id}
                                    grid_locations_dict[segment_id] = grid_target
                                    max_inst_point_num_in_grid = grid_num_inst_points

                        # If no instance was covered the box is still valid, but label should be the panoptic category
                        if max_inst_point_num_in_grid == 0 and points_in_segment.any():
                            label_nums = np.bincount(labels[points_in_segment * (labels != self.config.data.ignore_label)])
                            if len(label_nums) > 0:
                                majority_label = label_nums.argmax()
                                grid_target = {'label': majority_label,
                                               'mask': np.zeros(coords.shape[0], dtype=bool),
                                               'grid_index': segment_id}
                                grid_locations_dict[segment_id] = grid_target

            multi_res_grid_locations_dict += [grid_locations_dict]

        # debugging
        # import open3d as o3d
        #
        # for i in range(grid_indices.shape[1]):
        #     rnd_colors = np.random.rand(200, 3)
        #     grid_colors = rnd_colors[grid_indices[:, i] % 200]
        #     grid_colors[grid_indices[:, i] == -1, :] = 0.
        #
        #     grid_cloud = o3d.geometry.PointCloud()
        #     grid_cloud.points = o3d.utility.Vector3dVector(coords)
        #     grid_cloud.colors = o3d.utility.Vector3dVector(grid_colors)
        #
        #     o3d.visualization.draw_geometries([grid_cloud])

        return cloud_instances, multi_res_grid_locations_dict, mapped_instance_ids, grid_indices

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
