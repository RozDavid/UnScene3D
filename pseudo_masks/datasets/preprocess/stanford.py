import copy
import glob
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append('../../..')

from utils.pc_utils import save_point_cloud
from utils.utils import mkdir_p
import argparse
import open3d as o3d
import time
import felzenszwalb_cpp
from itertools import zip_longest
from scipy.spatial import KDTree
import random


STANFORD_3D_IN_PATH = '/cluster/himring/drozenberszki/Datasets/S3DIS/Stanford3dDataset_v1.2'
STANFORD_3D_OUT_PATH = '/cluster/himring/drozenberszki/Datasets/S3DIS/processed_s5000'

STANFORD_3D_TO_SEGCLOUD_LABEL = {
    4: 0,
    8: 1,
    12: 2,
    1: 3,
    6: 4,
    13: 5,
    7: 6,
    5: 7,
    11: 8,
    3: 9,
    9: 10,
    2: 11,
    0: 12,
}

class Stanford3DDatasetConverter:
    CLASSES = [
        'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
        'stairs', 'table', 'wall', 'window'
    ]

    class_map = {
        'ceiling': 0,
        'floor': 1,
        'wall': 2,
        'beam': 3,
        'column': 4,
        'window': 5,
        'door': 6,
        'table': 7,
        'chair': 8,
        'sofa': 9,
        'bookcase': 10,
        'board': 11,
        'clutter': 12,
        'stairs': 12  # stairs are also mapped to clutter
    }


    TRAIN_TEXT = 'train'
    VAL_TEXT = 'val'
    TEST_TEXT = 'test'

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

    @classmethod
    def create_poisson_felzenswalb_oversegmentation(cls, points, colors, config, mesh_file):

        # Estimate normals
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(points)
        scene_cloud.colors = o3d.utility.Vector3dVector(colors / 255.)
        scene_cloud.estimate_normals()

        if os.path.isfile(mesh_file):
            filtered_mesh = o3d.io.read_triangle_mesh(mesh_file)
        else:

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(scene_cloud, depth=9)

                # Filter based on densities
                densities = np.asarray(densities)

            # Remove faces with low density scores
            vertices_to_remove = densities < np.quantile(densities, config.density_threshold)
            filtered_mesh = copy.copy(mesh)
            filtered_mesh.remove_vertices_by_mask(vertices_to_remove)
            filtered_mesh.compute_vertex_normals()

        # Parse to numpy again
        vertices = np.array(filtered_mesh.vertices).astype(np.single)
        colors = np.array(filtered_mesh.vertex_colors).astype(np.single)
        faces = np.array(filtered_mesh.triangles).astype(np.intc)
        estimated_normals = np.array(filtered_mesh.vertex_normals).astype(np.single)

        # Compute the segmentation
        comps, connectivity = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, 0.005, config.min_vert_num)  # orig was min_vert_num=50

        # Filter out small segments and floaters
        mesh_points = np.array(filtered_mesh.vertices)
        vertices_tree = KDTree(mesh_points)
        segment_ids, segment_counts = np.unique(comps, return_counts=True)

        filtered_comps = comps.copy()
        for segment_id, segment_count in zip(segment_ids, segment_counts):
            
            if (segment_id not in connectivity) or (segment_count < config.min_vert_num):
                tmp = segment_id.copy()
                _, closest_point_ids = vertices_tree.query(mesh_points[comps == segment_id][0], k=segment_count+1)
                target_segment_id = comps[closest_point_ids][np.nonzero(comps[closest_point_ids] - tmp)[0][0]]
        
                # update at location 
                filtered_comps[comps == segment_id] = target_segment_id

        # Associate each point to a segment
        kdtree = KDTree(vertices)
        _, idx = kdtree.query(points)
        seg_labels = filtered_comps[idx]

        return filtered_mesh, seg_labels, connectivity



    @classmethod
    def convert_to_ply(cls, root_path, out_path, config):
        """Convert Stanford3DDataset to PLY format that is compatible with
        Synthia dataset. Assumes file structure as given by the dataset.
        Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
        """

        mode_name = config.mode_name
        txtfiles = glob.glob(os.path.join(root_path, '*/*.txt'))
        # Shuffle the files.
        random.shuffle(txtfiles)
        for txtfile in tqdm(txtfiles):
            file_sp = os.path.normpath(txtfile).split(os.path.sep)
            target_path = os.path.join(out_path, file_sp[-3])
            out_file = os.path.join(target_path, file_sp[-2] + '.ply')
            mesh_file = os.path.join(target_path, file_sp[-2] + '_mesh.ply')
            if os.path.exists(out_file):
                print(out_file, ' exists')
                continue

            annotation, _ = os.path.split(txtfile)
            subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
            coords, feats, labels, instances = [], [], [], []
            for inst, subcloud in enumerate(subclouds):
                # Read ply file and parse its rgb values.
                xyz, rgb = cls.read_txt(subcloud)
                _, annotation_subfile = os.path.split(subcloud)
                clsidx = cls.class_map[annotation_subfile.split('_')[0]]
                coords.append(xyz)
                feats.append(rgb)
                labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)
                instances.append(np.ones((len(xyz), 1), dtype=np.int32) * (inst + 1))


            if len(coords) == 0:
                print(txtfile, ' has 0 files.')
            else:
                # Concat
                coords = np.concatenate(coords, 0)
                feats = np.concatenate(feats, 0)
                labels = np.concatenate(labels, 0)
                instances = np.concatenate(instances, 0)
                # inds, collabels = ME.utils.sparse_quantize(
                #     coords,
                #     feats,
                #     labels,
                #     return_index=True,
                #     ignore_label=255,
                #     quantization_size=0.01  # 1cm
                # )
                # pointcloud = np.concatenate((coords[inds], feats[inds], collabels[:, None]), axis=1)

                # Generate Mesh and do Felzenswalb segmentation

                # Wrap in try catch
                mesh, comps, connectivity = cls.create_poisson_felzenswalb_oversegmentation(coords, feats, config, mesh_file=mesh_file)
                
                # try:
                #     mesh, comps, connectivity = cls.create_poisson_felzenswalb_oversegmentation(coords, feats, config)
                # except:
                #     print('Error in ', txtfile)
                #     continue

                pointcloud = np.concatenate((coords, feats, labels, instances, comps[:, None]), axis=1)

                # Write ply file.
                mkdir_p(target_path)
                save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)

                # Also save connectivity
                connectivity_file = os.path.join(target_path, file_sp[-2] + '_connectivity.npy')
                np.save(connectivity_file, connectivity)

                # Also save mesh
                o3d.io.write_triangle_mesh(mesh_file, mesh)


def generate_splits(stanford_out_path):
    """Takes preprocessed out path and generate txt files"""
    for i in range(1, 7):
        curr_path = os.path.join(stanford_out_path, f'Area_{i}')
        files = glob.glob(os.path.join(curr_path, '*.ply'))

        # Drop all files that contain the word 'mesh'
        files = [f for f in files if 'mesh' not in f]

        files = [os.path.relpath(full_path, stanford_out_path) for full_path in files]
        out_txt = os.path.join(stanford_out_path, f'area{i}.txt')
        with open(out_txt, 'w') as f:
            f.write('\n'.join(files))


if __name__ == '__main__':

    # Add argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_name', type=str, default='Area_1', help='which area to process')
    parser.add_argument('--min_vert_num', type=int, default=2000, help='path to output data')
    parser.add_argument('--density_threshold', type=float, default=0.1, help='At Poisson mesh should be filtered')
    config = parser.parse_args()

    Stanford3DDatasetConverter.convert_to_ply(os.path.join(STANFORD_3D_IN_PATH, config.mode_name), STANFORD_3D_OUT_PATH, config)
    # generate_splits(STANFORD_3D_OUT_PATH)
