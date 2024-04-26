import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
from scipy.spatial import KDTree

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

class FreeMaskPreprocessing(BasePreprocessing):

    FREEMASK_CLASS_IDS = (0, 1)
    FREEMASK_CLASS_NAMES = ('background', 'foreground')
    FREEMASK_COLOR_MAP = {0: (0, 0, 0), 1: (0, 0, 128)}
    FREEMASK_ORACLE_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(
            self,
            data_dir: str = "/canis/Datasets/ScanNet/public/v2",
            save_dir: str = "./data/processed/freemask",
            modes: tuple = ("train", "validation"),
            n_jobs: int = -1,
            git_repo: str = "/cluster/himring/drozenberszki/Datasets/ScanNet/ScanNet",
            oracle: bool = False,
            freemask_dir: str = "/mnt/data/Datasets/ScanNetFreeMask"):

        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.oracle = oracle
        git_repo = Path(git_repo)
        self.create_label_database(git_repo)
        self.freemask_base_path = Path(freemask_dir)

        for mode in self.modes:
            trainval_split_dir = git_repo / "Tasks" / "Benchmark"
            scannet_special_mode = "val" if mode == "validation" else mode
            with open(
                    trainval_split_dir / (f"scannetv2_{scannet_special_mode}.txt")
            ) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            scans_folder = "scans_test" if mode == "test" else "scans"
            filepaths = []
            for scene in split_file:
                filepaths.append(
                    self.data_dir / scans_folder / scene / (scene + "_vh_clean_2.ply")
                )
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self, git_repo):
        label_database = {}
        for row_id, class_id in enumerate(self.FREEMASK_CLASS_IDS):
            label_database[class_id] = {
                'color': self.FREEMASK_COLOR_MAP[class_id],
                'name': self.FREEMASK_CLASS_NAMES[row_id],
                'validation': True
            }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

       The first part is analogous to the scannet200 preprocessing.
       In second part we load the freemask data and masks proposals, which we assign to the points.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene, sub_scene = self._parse_scene_subscene(filepath.name)
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": sub_scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))

        # getting scene information
        description_filepath = Path(filepath).parent / filepath.name.replace(
            "_vh_clean_2.ply", ".txt"
        )
        with open(description_filepath) as f:
            scene_type = f.read().split("\n")[:-1]
        scene_type = scene_type[-1].split(" = ")[1]
        filebase["scene_type"] = scene_type
        filebase["raw_description_filepath"] = description_filepath

        # getting instance info
        instance_info_filepath = next(
            Path(filepath).parent.glob("*.aggregation.json")
        )
        segment_indexes_filepath = next(
            Path(filepath).parent.glob("*[0-9].segs.json")
        )
        instance_db = self._read_json(instance_info_filepath)
        segments = self._read_json(segment_indexes_filepath)
        segments = np.array(segments["segIndices"])
        filebase["raw_instance_filepath"] = instance_info_filepath
        filebase["raw_segmentation_filepath"] = segment_indexes_filepath

        # add segment id as additional feature
        segment_ids = np.unique(segments, return_inverse=True)[1]
        points = np.hstack((points, segment_ids[..., None]))

        # reading labels file
        label_filepath = filepath.parent / filepath.name.replace(
            ".ply", ".labels.ply"
        )
        filebase["raw_label_filepath"] = label_filepath
        label_coords, label_colors, labels = load_ply_with_normals(label_filepath)
        if not np.allclose(coords, label_coords):
            raise ValueError("files doesn't have same coordinates")

        # adding instance label
        labels = labels[:, np.newaxis]
        empty_instance_label = np.full(labels.shape, -1)
        labels = np.hstack((labels, empty_instance_label))
        for instance in instance_db["segGroups"]:
            segments_occupied = np.array(instance["segments"])
            occupied_indices = np.isin(segments, segments_occupied)
            labels[occupied_indices, 1] = instance["id"]

        # Mask labels to only foreground for all labels except wall and floor (which should be the ideal case)
        foregorund_labels = np.isin(labels[:, 0], self.FREEMASK_ORACLE_CLASS_IDS)
        labels[foregorund_labels, 0] = 1

        # Mask out all instances which are associated to ignored labels
        labels[~foregorund_labels, 1] = -1
        points = np.hstack((points, labels))

        # Adding benchmark style instance labels - gt labels are all 1 or 0
        gt_data = points[:, -2] * 1000 + points[:, -1] + 1

        processed_filepath = self.save_dir / mode / f"{scene:04}_{sub_scene:02}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"scene{scene:04}_{sub_scene:02}.txt"
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

        # Align coordinates with axis aligned freemask coords
        info_dict = {}
        with open(description_filepath) as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for line in f:
                    (key, val) = line.split(" = ")
                    info_dict[key] = np.fromstring(val, sep=' ')
        axis_alignment = info_dict['axisAlignment'].reshape(4, 4) if 'axisAlignment' in info_dict else np.identity(4)

        # Also rotate the orig scene with it
        homo_coords = np.hstack((points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)))
        aligned_points = homo_coords @ axis_alignment.T[:, :3]

        # Load freemask proposals if not oracle
        if self.oracle:
            instance_num = len(set(np.unique(labels[:, 1])) - {-1})
            aligned_freemask_masks = np.zeros((aligned_points.shape[0], instance_num), dtype=np.float32)
            for i in range(instance_num):
                aligned_freemask_masks[labels[:, 1] == i, i] = 1.0
        else:

            # Load coords of the freemask proposals
            try:
                freemask_scene_cloud = np.load(self.freemask_base_path / f"train/scene{scene:04}_{sub_scene:02}_cloud.npy")
                freemask_coords = freemask_scene_cloud[:, :3]
            except:
                print(f"Could not load cloud for scene {scene:04}_{sub_scene:02}")
                return None

            try:
                freemask_masks = np.load(self.freemask_base_path / f"train/scene{scene:04}_{sub_scene:02}_masks.npy")
            except:
                print(f"Could not load freemask masks for scene {scene:04}_{sub_scene:02}")
                return None

            # Be sure that the order of the points is unchanged, so rather find the closest for all
            freemask_tree = KDTree(freemask_coords)  # for rounding shift at 2cm resolution
            _, lr_hr_matches = freemask_tree.query(aligned_points, k=1)

            aligned_freemask_masks = freemask_masks[lr_hr_matches]

        # Save mask data
        freemask_filepath = self.save_dir / mode / f"{scene:04}_{sub_scene:02}_freemasks.npy"
        np.save(freemask_filepath, aligned_freemask_masks.astype(np.float32))
        filebase["freemask_filepath"] = str(freemask_filepath)

        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/freemask/train_database.yaml"
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
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(FreeMaskPreprocessing)
