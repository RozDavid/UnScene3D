import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
from scipy.spatial import KDTree
import open3d as o3d

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

class FreeMaskPreprocessing(BasePreprocessing):

    FREEMASK_CLASS_IDS = (0, 1)
    FREEMASK_CLASS_NAMES = ('background', 'foreground')
    FREEMASK_COLOR_MAP = {0: (0, 0, 0), 1: (0, 0, 128)}

    def __init__(
            self,
            data_dir: str = "/cluster/himring/drozenberszki/Datasets/ArKitScenes",
            save_dir: str = "/cluster/himring/drozenberszki/Datasets/Mask3D/data/processed/unscene3d_arkit",
            modes: tuple = ("train", "validation"),
            n_jobs: int = 8,
            freemask_dir: str = "/cluster/himring/drozenberszki/Datasets/ArKitScenes"):

        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.create_label_database()
        self.freemask_base_path = Path(freemask_dir)

        for mode in self.modes:
            trainval_split_dir = data_dir / Path("split")
            with open(trainval_split_dir / f"{mode}.txt") as f:
                split_file = f.read().split("\n")[:-1]

            scans_folder = "freemask"
            filepaths = []
            for scene in split_file:
                filepaths.append(self.data_dir / scans_folder / f'{scene}_cloud.npy')
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = {}
        for row_id, class_id in enumerate(self.FREEMASK_CLASS_IDS):
            label_database[class_id] = {
                'color': self.FREEMASK_COLOR_MAP[class_id],
                'name': self.FREEMASK_CLASS_NAMES[row_id],
                'validation': True
            }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def load_ply_cloud_with_normals(self, filepath):

        # load cloud
        cloud = o3d.io.read_point_cloud(str(filepath))

        # estimate normals
        cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        vertices = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        feats = np.asarray(cloud.colors)
        feats = np.hstack((feats, normals))
        labels = np.zeros(len(vertices), dtype=np.int32)

        return vertices, feats, labels

    def process_file(self, filepath, mode):

        scene, sub_scene = self._parse_scene_subscene(filepath.name)
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": sub_scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        try:
            freemask_masks = np.load(self.freemask_base_path / f"freemask/{scene}_masks.npy")
            cloud_np = np.load(filepath)
        except:
            print(f"Could not load freemask masks for scene {scene}")
            return None

        coords = cloud_np[:, :3]
        features = cloud_np[:, 3:6]
        labels = cloud_np[:, 7][:, None]
        instances = cloud_np[:, 8][:, None]
        labels = np.hstack((labels, instances))
        segment_ids = cloud_np[:, -1]

        # Estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        features = np.hstack((features, normals))

        file_len = len(coords)
        filebase["file_len"] = file_len

        # Setting placeholder metadata which is not present for Arkit
        filebase["scene_type"] = 'room'
        filebase["raw_description_filepath"] = ''
        filebase["raw_instance_filepath"] = ''
        filebase["raw_segmentation_filepath"] = ''

        # add segment id as additional feature
        points = np.hstack((coords, features))
        points = np.hstack((points, segment_ids[..., None]))
        points = np.hstack((points, labels))

        # Adding benchmark style instance labels - gt labels are all 1 or 0
        gt_data = points[:, -2] * 1000 + points[:, -1] + 1

        processed_filepath = self.save_dir / mode / f"{scene}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"scene{scene}.txt"
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean()),
        ]

        # Save mask data
        freemask_filepath = self.save_dir / mode / f"{scene}_freemasks.npy"
        np.save(freemask_filepath, freemask_masks.astype(np.float32))
        filebase["freemask_filepath"] = str(freemask_filepath)

        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/arkit/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    def _parse_scene_subscene(self, name):
        return name.split("_")[0], 0


if __name__ == "__main__":
    Fire(FreeMaskPreprocessing)
