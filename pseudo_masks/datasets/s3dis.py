import logging
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import spatial
from plyfile import PlyData
from lib.utils.utils import read_txt, fast_hist, per_class_iu, Timer
from lib.datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type, cache
import lib.utils.transforms as t
import MinkowskiEngine as ME

import open3d as o3d


class StanfordVoxelizationDatasetBase:
    CLIP_SIZE = None
    CLIP_BOUND = None
    LOCFEAT_IDX = 2
    ROTATION_AXIS = 'z'
    NUM_LABELS = 14
    IGNORE_LABELS = (10,)  # remove stairs, following SegCloud
    # CLASSES = [
    #     'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
    #     'table', 'wall', 'window'
    # ]
    IS_FULL_POINTCLOUD_EVAL = True
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train.txt',
        DatasetPhase.Val: 'val.txt',
        DatasetPhase.TrainVal: 'trainval.txt',
        DatasetPhase.Test: 'test.txt'
    }

    def _augment_coords_to_feats(self, coords, feats, labels=None):
        # Center x,y
        coords_center = coords.mean(0, keepdims=True)
        coords_center[0, 2] = 0
        norm_coords = coords - coords_center
        feats = np.concatenate((feats, norm_coords), 1)
        return coords, feats, labels


class StanfordDataset(StanfordVoxelizationDatasetBase, VoxelizationDataset):

    DATA_PATH_FILE = {
        DatasetPhase.Train: ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area6.txt'],
        DatasetPhase.TrainVal: ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area5.txt', 'area6.txt'],
        DatasetPhase.Val: 'area5.txt',
        DatasetPhase.Test: 'area5.txt'
    }

    # Voxelization arguments
    VOXEL_SIZE = 0.05  # 5cm
    CLIP_BOUND = 4  # [-N, N]
    TEST_CLIP_BOUND = None
    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = \
        ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))
    AUGMENT_COORDS_TO_FEATS = True
    NUM_IN_CHANNEL = 6


    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 augment_data=True,
                 elastic_distortion=False,
                 phase=DatasetPhase.Train):

        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_root = config.data.scannet_path
        if isinstance(self.DATA_PATH_FILE[phase], (list, tuple)):
            data_paths = []
            for split in self.DATA_PATH_FILE[phase]:
                data_paths += read_txt(os.path.join(data_root, split))
        else:
            data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
        logging.info('Loading {} {}: {}'.format(self.__class__.__name__, phase,
                                                self.DATA_PATH_FILE[phase]))

        self.data_paths = data_paths
        self.scene_names = [data_path.replace('/', '_').replace('.ply', '') for data_path in self.data_paths]
        logging.info('Loading {} scenes into {} for {} phase'.format(len(self.data_paths), self.__class__.__name__, phase))

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

    @cache
    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        instances = np.array(data['instance_id'], dtype=np.int32)
        segment_ids = np.array(data['segment_id'], dtype=np.int32)
        scene_name = self.scene_names[index]

        # Push coords to origin
        coords -= coords.min(0, keepdims=True)

        # Load also segment connectivity
        segment_connectivity_path = filepath.parent / (filepath.stem + '_connectivity.npy')
        seg_connectivity = np.load(segment_connectivity_path)

        return coords, feats, labels, instances, segment_ids, seg_connectivity, scene_name

    def prepare_scene_data(self, coords, feats, labels, instance_ids, segment_ids):

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

        feats = feats / 255. - 0.5  # normalize_color

        return coords, feats, labels, instance_ids, segment_ids, transformations

    def get_full_cloud_by_scene_name(self, scene_name):

        # get index from scene name
        index = self.data_paths.index(scene_name[:6] + '/' + scene_name[7:] + '.ply')

        coords, feats, labels, instances, segment_ids, seg_connectivity, scene_name = self.load_ply(index)
        return coords, feats, labels, instances, segment_ids, seg_connectivity, scene_name

    def __getitem__(self, index):

        # Load data
        coords, feats, labels, instance_ids, segment_ids, seg_connectivity, scene_name = self.load_ply(index)
        images, camera_poses, color_intrinsics = None, None, None

        # Voxelize and augment
        coords, feats, labels, instance_ids, segment_ids, transformations = self.prepare_scene_data(coords, feats, labels, instance_ids, segment_ids)

        # Collect in tuple
        return_args = (coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids[:, None], [seg_connectivity], transformations[1].astype(np.float32))

        return return_args


class StanfordArea5Dataset(StanfordDataset):
    DATA_PATH_FILE = {
        DatasetPhase.Train: ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area6.txt'],
        DatasetPhase.Val: 'area5.txt',
        DatasetPhase.Test: 'area5.txt'
    }


class Stanford2cmDataset(StanfordDataset):
    VOXEL_SIZE = 0.02  # 2cm


def test(config):
    """Test point cloud data loader.
    """
    from torch.utils.data import DataLoader
    import open3d as o3d
    def make_pcd(coords, feats):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[:, :3].float().numpy())
        pcd.colors = o3d.utility.Vector3dVector(feats[:, :3].numpy() / 255)
        return pcd

    timer = Timer()
    DatasetClass = StanfordArea5Dataset
    transformations = [
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.ChromaticAutoContrast(),
        t.ChromaticTranslation(config.data_aug_color_trans_ratio),
        t.ChromaticJitter(config.data_aug_color_jitter_std),
    ]
    dataset = DatasetClass(
        config,
        prevoxel_transform=t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS),
        input_transform=t.Compose(transformations),
        augment_data=True,
        cache=True,
        elastic_distortion=True)
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=t.cfl_collate_fn_factory(limit_numpoints=False),
        batch_size=1,
        shuffle=True)
    # Start from index 1
    iter = data_loader.__iter__()
    for i in range(100):
        timer.tic()
        coords, feats, labels = iter.next()
        pcd = make_pcd(coords, feats)
        o3d.visualization.draw_geometries([pcd])
        print(timer.toc())


if __name__ == '__main__':
    from config import get_config

    config = get_config()
    test(config)
