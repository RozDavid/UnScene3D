import random
import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from datasets.semseg import *
import felzenszwalb_cpp

logger = logging.getLogger(__name__)

class SemanticSegmentationFreeDataset(Dataset):
    """Docstring for SemanticSegmentationDataset. """

    def __init__(
        self,
        dataset_name="freemask",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet_freemask",
        label_db_filepath: Optional[str] = "data/processed/scannet_freemask/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = 2,
        data_percent: Optional[float] = 1.0,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        max_cut_region=0,
        point_per_cut=100,
        flip_in_center=False,
        noise_rate=0.0,
        resample_points=0.0,
        cache_data=False,
        task="instance_segmentation",
        is_tta=False,
        reps_per_epoch=1,
        area=-1,
        on_crops=False,
        eval_inner_core=-1,
        add_clip=False,
        is_elastic_distortion=True,
        color_drop=0.0,
        freemask_hard_threshold=0.5,
        freemask_extent_max_ratio=0.8,
        max_num_gt_instances=-1,
        load_self_train_data=False,
        self_train_data_dir=None,
        num_self_train_data=5,
        resegment_mesh=False,
        segment_min_vert_num=20,
    ):

        assert task in ["instance_segmentation"], "unknown task"

        self.add_clip = add_clip
        self.dataset_name = dataset_name
        self.is_elastic_distortion = is_elastic_distortion
        self.color_drop = color_drop

        # Freemask color map
        self.color_map = {
            0: [0, 255, 0],  # background
            1: [170, 120, 200]}  # foreground

        self.task = task

        self.area = area
        self.eval_inner_core = eval_inner_core

        self.reps_per_epoch = reps_per_epoch

        self.is_tta = is_tta
        self.on_crops = on_crops

        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [Path(self.data_dir)]
        self.ignore_label = ignore_label
        self.label_offset = 0
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.max_cut_region = max_cut_region
        self.point_per_cut = point_per_cut
        self.flip_in_center = flip_in_center
        self.noise_rate = noise_rate
        self.resample_points = resample_points
        self.freemask_hard_threshold = freemask_hard_threshold
        self.freemask_extent_max_ratio = freemask_extent_max_ratio
        self.max_num_gt_instances = max_num_gt_instances

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            if not (database_path / f"{mode}_database.yaml").exists():
                print(f"generate {database_path}/{mode}_database.yaml first")
                exit()
            self._data.extend(SemanticSegmentationDataset._load_yaml(database_path / f"{mode}_database.yaml"))

        # Remove invalid data
        self._data = [data for data in self._data if data is not None and 'filepath' in data]

        if data_percent < 1.0:
            self._data = sample(self._data, int(len(self._data) * data_percent))
        labels = SemanticSegmentationDataset._load_yaml(Path(label_db_filepath))
        self._labels = SemanticSegmentationDataset._select_correct_labels(labels, num_labels)

        # normalize color channels
        if Path(str(color_mean_std)).exists():
            color_mean_std = SemanticSegmentationDataset._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error("pass mean and std as tuple of tuples, or as an .yaml file")

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        self.cache_data = cache_data
        # new_data = []
        if self.cache_data:
            new_data = []
            for i in range(len(self._data)):
                point_filepath = self.data[i]["filepath"].replace("../../", "")
                masks_filepath = point_filepath.replace(".npy", "_freemasks.npy")
                self._data[i]['data'] = np.load(point_filepath)
                self._data[i]['freemask'] = np.load(masks_filepath)

        # Add self training arguments
        self.load_self_train_data = load_self_train_data
        self.self_train_data_dir = self_train_data_dir
        self.num_self_train_data = num_self_train_data

        # If we want to have our own oversegmentation
        self.resegment_mesh = resegment_mesh
        self.segment_min_vert_num = segment_min_vert_num

    @property
    def data(self):
        """ database file containing information about preproscessed dataset """
        return self._data

    @property
    def label_info(self):
        """ database file containing information labels used by dataset """
        return self._labels

    def __len__(self):
        if self.is_tta:
            return 5*len(self.data)
        else:
            return self.reps_per_epoch*len(self.data)

    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)


    def segment_mesh(self, points, mesh_fpath):

        # For arkit scenes quick fix
        if ".npy" in mesh_fpath:
            parent = Path(mesh_fpath).parent
            name = Path(mesh_fpath).stem.split("_")[0]
            mesh_fpath = str(parent / '../meshes/low_res' / f"{name}.ply")

        # Load mesh with open3d
        scene_mesh = o3d.io.read_triangle_mesh(mesh_fpath)

        if len(np.array(scene_mesh.vertices)) == 0:
            print(f"Can't segment mesh as it is empty at {mesh_fpath}")
            return None

        # Parse segment to correct format
        # Parse to correct type and call algorithm
        vertices = np.array(scene_mesh.vertices).astype(np.single)
        colors = np.array(scene_mesh.vertex_colors).astype(np.single)
        faces = np.array(scene_mesh.triangles).astype(np.intc)
        seg_indices, seg_connectivity = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, 0.005, self.segment_min_vert_num)

        # We have to match the mesh vertices to the points
        if points.shape[0] != vertices.shape[0]:
            # Find KNNs
            tree = KDTree(vertices)
            dist, idx = tree.query(points, k=1)
            seg_indices = seg_indices[idx.flatten()]

        return seg_indices, seg_connectivity


    def load_self_train_masks(self, idx, points, freemasks, drop_original=False):

        # Load self training data
        scene_id = Path(self.data[idx]["filepath"]).stem
        try:
            self_train_fname = 'masks' if self.load_self_train_data != 'refined' else 'masks_refined'
            self_train_cloud = np.load(f'{self.self_train_data_dir}/freemasks/scene{scene_id}_cloud.npy')
            self_train_masks = np.load(f'{self.self_train_data_dir}/freemasks/scene{scene_id}_{self_train_fname}.npy')
        except FileNotFoundError:
            print(f'Could not load self training data for scene{scene_id}')
            return freemasks

        # Check if coords are the same (all close), if not we have to find KNNs
        if len(points) != len(self_train_cloud) or not np.allclose(points[:, :3], self_train_cloud[:, :3]):
            # Find KNNs
            self_train_points = self_train_cloud[:, :3]
            tree = KDTree(self_train_points)
            dist, ind = tree.query(points, k=1)
            self_train_masks = self_train_masks[ind]

        if drop_original:  # if we want to only keep the new masks from the previous iteration
            freemasks = np.zeros((freemasks.shape[0], 0), dtype=freemasks.dtype)

        # Check if IoU of new instance is larger than 0.5
        foreground_insts = np.any(freemasks, axis=1)

        # Iterate over the self train instances and add if condition is met
        num_added_insts, self_train_id = 0, 0
        while num_added_insts < self.num_self_train_data and self_train_id < self_train_masks.shape[1]:
            new_inst = self_train_masks[:, self_train_id].astype(bool)
            useful_iou = np.logical_and(~foreground_insts, new_inst).sum() / new_inst.sum()
            if useful_iou > 0.5:

                # Clean new inst and only add the part which was not covered by the previous instances
                new_inst = np.logical_and(~foreground_insts, new_inst)

                freemasks = np.concatenate([freemasks, new_inst.reshape(-1, 1)], axis=1)
                foreground_insts = np.any(freemasks, axis=1)
                num_added_insts += 1
            self_train_id += 1

        return freemasks

    def __getitem__(self, idx: int):
        idx = idx % len(self.data)

        # Load points and masks
        if self.cache_data:
            points = self.data[idx]['data']
            freemasks = self.data[idx]['freemask']
        else:
            assert not self.on_crops, "you need caching if on crops"
            if self.data[idx]["filepath"] is not None:
                point_filepath = self.data[idx]["filepath"].replace("../../", "")
                masks_filepath = point_filepath.replace(".npy", "_freemasks.npy")
                points = np.load(point_filepath)
                freemasks = np.load(masks_filepath)
            else:  # this means there was no mask for this point cloud
                return self.__getitem__(np.random.randint(len(self.data)))

        # Load additional train instances besides the UnScene/FreeMask ones
        if self.load_self_train_data:
            freemasks = self.load_self_train_masks(idx, points, freemasks)

        # Filter out gt masks if we want to have the most confident ones only
        if self.max_num_gt_instances > 0:
            freemasks = freemasks[:, :self.max_num_gt_instances]

        # Load scene data and drop instances and labels
        coordinates, color, normals, segments, _, segment_connectivity = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
            []
        )

        if self.resegment_mesh:
            segments, segment_connectivity = self.segment_mesh(coordinates, self.data[idx]["raw_filepath"])

        # Filter freemasks based on size compared to full extent of the scene
        scene_extent = (np.max(coordinates, axis=0) - np.min(coordinates, axis=0))[:2] * self.freemask_extent_max_ratio  # we only care about XY
        keep_masks = []
        for freemask_id in range(freemasks.shape[1]):
            mask = freemasks[:, freemask_id] > self.freemask_hard_threshold
            if mask.sum() == 0:
                continue  # Don't keep it for empty arrays
            mask_extents = (np.max(coordinates[mask], axis=0) - np.min(coordinates[mask], axis=0))[:2]
            if not np.any(mask_extents > scene_extent):
                keep_masks += [freemask_id]
        freemasks = freemasks[:, keep_masks]

        # If it is empty return a random sample
        if len(keep_masks) == 0:
            return self.__getitem__(np.random.randint(len(self.data) - 1))

        # Calculate labels, where at least one mask is kept, append that to beginning of freemasks
        labels = np.any(freemasks > self.freemask_hard_threshold, axis=1).astype(np.int32)
        freemasks = (freemasks > self.freemask_hard_threshold).astype(np.int32)
        freemasks = np.concatenate([labels.reshape(-1, 1), freemasks], axis=1)

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode and hasattr(self.volume_augmentations, 'transforms'):
            coordinates -= coordinates.mean(0)

            try:
                coordinates += np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            # Apply geometric augmentations
            if self.flip_in_center:
                coordinates = flip_in_center(coordinates)

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            if random() < 0.95:
                if self.is_elastic_distortion:
                    for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                        coordinates = elastic_distortion(
                            coordinates, granularity, magnitude
                        )
            aug = self.volume_augmentations(
                points=coordinates, normals=normals, features=color, labels=freemasks,
            )
            coordinates, color, normals, freemasks = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )

            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(self.image_augmentations(image=pseudo_image)["image"])

            if self.point_per_cut != 0:
                number_of_cuts = int(len(coordinates) / self.point_per_cut)
                for _ in range(number_of_cuts):
                    size_of_cut = np.random.uniform(0.05, self.max_cut_region)
                    # not wall, floor or empty
                    point = choice(coordinates)
                    x_min = point[0] - size_of_cut
                    x_max = x_min + size_of_cut
                    y_min = point[1] - size_of_cut
                    y_max = y_min + size_of_cut
                    z_min = point[2] - size_of_cut
                    z_max = z_min + size_of_cut
                    indexes = crop(
                        coordinates, x_min, y_min, z_min, x_max, y_max, z_max
                    )
                    coordinates, normals, color, labels = (
                        coordinates[~indexes],
                        normals[~indexes],
                        color[~indexes],
                        labels[~indexes],
                    )

            if (self.resample_points > 0) or (self.noise_rate > 0):
                coordinates, color, normals, freemasks = random_around_points(
                    coordinates,
                    color,
                    normals,
                    freemasks,
                    self.resample_points,
                    self.noise_rate,
                    self.ignore_label,
                )

            if random() < self.color_drop:
                color[:] = 255

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        # if self.task != "semantic_segmentation":
        if self.data[idx]['raw_filepath'].split("/")[-2] in ['scene0636_00', 'scene0154_00']:
            return self.__getitem__(0)

        # Append segments to masks to resemble return shape to original function
        freemasks = np.hstack((freemasks, segments[..., None].astype(freemasks.dtype))).astype(np.int32)

        if 'arkit' in self.data[idx]['raw_filepath'].lower():
            scene_name = f"scene{self.data[idx]['raw_filepath'].split('/')[-1].split('_')[0]}"
            # features[:, :3], raw_color = features[:, :3] * 0., raw_color * 0.
        else:
            scene_name = self.data[idx]['raw_filepath'].split('/')[-2]

        return coordinates, features, freemasks, scene_name,  raw_color, raw_normals, raw_coordinates, idx, segment_connectivity

    def _remap_model_output(self, output):
        # Here an identity function only, but can be used to remap the output of the model for more categories
        output = np.array(output)
        return output.copy()
