import os.path
import hydra
import logging
from omegaconf import OmegaConf

import sys
sys.path.append('..')

from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from models import load_model
from utils.utils import load_state_with_same_shape
from utils.freemask_utils import *
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
import networkx as nx
from scipy.spatial import KDTree
import contextlib
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm
import pyviz3d.visualizer as vis

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

## Util functions
def crop_region(points, cluster_label, curr_cluster, growth_region=0.):
    cls_points = np.where(cluster_label == curr_cluster)[0]

    cluster_center = points[cls_points].mean(axis=0)

    # center_dists = np.sqrt(np.sum((points - cluster_center)**2, axis=-1))
    # farthest_point = np.argmax(center_dists[cls_points])

    # look for points around the cluster with radius = farthest_dist + growth_region
    max_xyz = np.max(np.abs(points[cls_points] - cluster_center), axis=0) + growth_region

    # drop wrong labels
    if np.sum(max_xyz > 300.):
        return np.zeros((len(points),)).astype(bool)

    upper_idx = np.sum((points[:, :3] < cluster_center + max_xyz).astype(np.int32), 1) == 3
    lower_idx = np.sum((points[:, :3] > cluster_center - max_xyz).astype(np.int32), 1) == 3

    return ((upper_idx) & (lower_idx))


def get_cluster_saliency(x, out, grad_point, corr_points=None):
    if corr_points is None:
        score = torch.mean(torch.mean(out.F[grad_point], dim=-1))
    else:
        score = torch.mean(torch.mean(out.F[corr_points][grad_point], dim=-1))
    x.F.retain_grad()
    score.backward(retain_graph=True)

    slc, _ = torch.max(torch.abs(x.F.grad), dim=-1)
    slc = torch.tanh((slc - slc.mean()) / slc.std())
    slc = slc.detach().numpy()

    return slc


def sample_source(source_seeds, points, tree, k=8):
    source = []
    for seed in source_seeds:
        _, neighbor_index = tree.query(points[seed], k=k)
        neighbor_index = neighbor_index.flatten()
        source += neighbor_index.tolist()

    return np.asarray(source)


def affinity(sim, sigma):
    return np.exp(-sim / (2 * (sigma ** 2)))


def manhattan(a, b):
    return np.sum(np.abs(a - b))


def array_to_sequence(batch_data):
    return [row for row in batch_data]


def array_to_torch_sequence(batch_data):
    return [torch.from_numpy(row).float() for row in batch_data]


def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None, device='cpu'):
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    return ME.TensorField(
        features=p_feats,
        coordinates=p_coord,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device=device,
    )


def build_graph(feats, slc, points, center, source_num, cluster_idx,
                sink_factor=2, source_factor=2,
                lambda_=0.1, omega_=10, sigma=1.0, k_neighbors=8):
    G = nx.Graph()
    G.add_node('source')
    G.add_node('sink')

    # build knn graph
    cluster_tree = KDTree(points)

    # Sample Seeds
    source_seeds = np.argsort(slc[:, 0])[-max(1, int(source_num / source_factor)):]
    source = sample_source(source_seeds, points, cluster_tree)

    # remove proposal points from background seeds if any
    sink_num = max(1, int((len(points) - source_num) / sink_factor))
    sink = np.argsort(slc[:, 0])[:sink_num]
    sink = sink[~np.in1d(sink, cluster_idx)]

    # define the points probs for foreground and background
    src_probs = np.ones((len(points),)) * 1e-20
    snk_probs = np.ones((len(points),)) * 1e-20

    # set the select seeds with high probability
    src_probs[np.clip(source, 0, len(src_probs) - 1)] = 1.
    snk_probs[np.clip(source, 0, len(snk_probs) - 1)] = 1.

    # build graph
    for i in range(len(points)):
        # edges between non-terminal and terminal vertex
        G.add_node(f'{i}')
        G.add_edge('source', f'{i}')
        G['source'][f'{i}']['capacity'] = -lambda_ * np.log(src_probs[i])

        G.add_edge(f'{i}', 'sink')
        G[f'{i}']['sink']['capacity'] = -lambda_ * np.log(snk_probs[i])

        # edges between non-terminal and non-terminal vertex
        _, neighbor_index = cluster_tree.query(points[i], k=k_neighbors)
        neighbor_index = np.clip(neighbor_index.flatten(), 0, len(points) - 1)

        # compute the edges weights between each point and its K neighbors
        for k in neighbor_index:
            G.add_edge(f'{i}', f'{k}')

            dissim_ik = manhattan(feats[i], feats[k])
            diff_ik = affinity(dissim_ik, sigma)
            G[f'{i}'][f'{k}']['capacity'] = omega_ * diff_ik

    return G

def graph_cut(G):
    _, partition = nx.minimum_cut(G, "source", "sink")
    reachable, non_reachable = partition

    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    ins_points = []
    for px in list(cutset):
        if px[1] == 'sink':
            continue
        ins_points.append(int(px[1]))

    return np.asarray(ins_points)

def handle_scene(index, dataset, model, output_base_path, z_max=0.25, roi_size = 0., refine=True):

    batch = dataset.__getitem__(index)
    coords, feats, target, instances, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, *transform = batch
    coords = np.concatenate((np.zeros((coords.shape[0], 1)), coords), axis=-1)
    coords, feats = torch.from_numpy(coords), torch.from_numpy(feats)
    segment_ids = segment_ids[:, -1]

    # Make target labels to freemask labels
    target[target > 1] = 1
    target[target < 1] = 0

    out_fname = os.path.join(output_base_path, f'outputs/eval_3duis/refined_preds_{scene_name}.npy')
    out_fname_comps = os.path.join(output_base_path, f'outputs/eval_3duis/hdbscan_{scene_name}.npy')
    out_fname_coords = os.path.join(output_base_path, f'outputs/eval_3duis/coords_{scene_name}.npy')
    out_fname_orig_segments = os.path.join(output_base_path, f'outputs/eval_3duis/segments_{scene_name}.npy')
    out_dirname_visualization = os.path.join(output_base_path, f'outputs/eval_3duis/visualization_{scene_name}')
    if os.path.exists(out_fname):
        print(f"Scene already processed {scene_name} with index {index}")
        pred_ins_full = np.load(out_fname)
        return pred_ins_full, index, target, instances
    else:

        print(f"Processing scene {scene_name} with index {index}")

        # Ground removal and clustering
        # Calculate normals of coords and get clusters
        vertices, colors = coords[:, 1:].numpy() * dataset.VOXEL_SIZE, feats.numpy() + feats.min().item()
        vertex_indexer = np.arange(len(vertices))
        dbscan_pcd = o3d.geometry.PointCloud()
        dbscan_pcd.points = o3d.utility.Vector3dVector(vertices)
        dbscan_pcd.colors = o3d.utility.Vector3dVector(colors)
        dbscan_pcd.estimate_normals()

        # Find actual floor inliers
        floor_mask = vertices[:, 2] < z_max
        floor_ids = vertex_indexer[floor_mask]
        floor_points = vertices[floor_mask]
        floor_colors = colors[floor_mask]

        floor_pcd = o3d.geometry.PointCloud()
        floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
        floor_pcd.colors = o3d.utility.Vector3dVector(floor_colors)

        # Plane segmentation
        plane_model, inliers = floor_pcd.segment_plane(distance_threshold=0.05,
                                                       ransac_n=3,
                                                       num_iterations=1000)

        floor_inliers = floor_ids[inliers]
        floor_outliers = np.ones(len(vertices), dtype=bool)
        floor_outliers[floor_inliers] = False

        # HDBSCAN clustering and add floor as cluster 1
        clusterer = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=1)
        clusterer.fit(np.concatenate((vertices, colors, np.array(dbscan_pcd.normals)), axis=1))
        comps = clusterer.labels_ + 2
        comps[floor_inliers] = 0
        unique_instances, num_pts = np.unique(comps, return_counts=True)

        # Remove floor and ignore
        unique_instances, num_pts = unique_instances[2:], num_pts[2:]

        # Sort from larger to smaller to keep always the minority instances
        sorted_segment_inds = num_pts.argsort()
        unique_instances = unique_instances[sorted_segment_inds[::-1]]
        num_pts = num_pts[sorted_segment_inds[::-1]]

        if refine:
            # The 3DUIS algorithm
            slc_full = np.zeros((len(vertices),), dtype=int)
            pred_ins_full = np.zeros((len(vertices),), dtype=int)

            for cluster in tqdm(unique_instances):  # the first one is floor, the second one is ignored labels

                # instance with id 0 is the ground ignore this and instance with just few point
                cls_points = np.where(comps == cluster)[0]
                if cluster == 0 or len(cls_points) <= 5:
                    continue

                # get cluster
                cluster_center = coords.float()[cls_points, 1:].mean(dim=0).numpy()

                # crop a ROI around the cluster
                window_points = crop_region(coords[:, 1:].numpy().astype(float), comps, cluster, roi_size)

                # skip when ROI is empty
                if not np.sum(window_points):
                    continue

                # get the closest point to the center
                center_dists = np.sqrt(np.sum((vertices[window_points] - cluster_center) ** 2, axis=-1))
                cluster_center = np.argmin(center_dists)

                # build input only with the ROI points
                input_features = numpy_to_sparse_tensor(vertices[window_points][np.newaxis, :, :],
                                                        colors[window_points][np.newaxis, :, :], device=model.device)
                input_features.F.requires_grad = True
                input_features_sparse = input_features.sparse()

                # Get deep features for input region
                _, feature_maps = model(input_features_sparse)
                out_features = feature_maps[f'res_1']
                out_features = out_features.slice(input_features)

                # compute saliency for the point in the center
                slc = get_cluster_saliency(input_features, out_features, np.where(comps[window_points] == cluster)[0])

                # place the computed saliency into the full point cloud for comparison
                slc_full[window_points] = np.maximum(slc_full[window_points], slc)

                # build graph representation
                G = build_graph(out_features.F.detach().numpy(),
                                slc[:, np.newaxis],
                                vertices[window_points],
                                cluster_center,
                                np.sum(comps == cluster),
                                np.where(comps[window_points] == cluster)[0], )

                # perform graph cut
                ins_points = graph_cut(G)

                # create point-wise prediction matrix
                pred_ins = np.zeros(input_features.F.shape[0]).astype(int)
                if len(ins_points) != 0:
                    pred_ins[ins_points] = cluster

                # Update the full set of predictions
                pred_ins_full[window_points] = np.maximum(pred_ins_full[window_points], pred_ins)

            print(f"Finished scene {scene_name} with index {index}")

            # Saved refined prediction
            np.save(out_fname, pred_ins_full)
            np.save(out_fname_comps, comps)
            np.save(out_fname_orig_segments, segment_ids)
            np.save(out_fname_coords, coords[:, 1:].numpy() * dataset.VOXEL_SIZE)

            # We also want to save the visualized segments
            v = vis.Visualizer()
            orig_points = coords[:, 1:].numpy() * dataset.VOXEL_SIZE
            v.add_points('color', orig_points,
                         colors=(colors) * 255., visible=True)

            # generate random colors for the predicted instances
            pred_colors = np.random.rand(np.unique(pred_ins_full).max() + 1, 3) * 255.
            v.add_points('preds', orig_points,
                         colors=pred_colors[pred_ins_full], visible=True)

            # generate random colors for the original comps
            orig_colors = np.random.rand(np.unique(comps).max() + 1, 3) * 255.
            v.add_points('dbscan',
                         orig_points,
                         colors=orig_colors[comps], visible=True)

            v.save(out_dirname_visualization)

            # return full res predictions
            return pred_ins_full, index, target, instances
        else:
            return comps, index, target, instances


@hydra.main(config_path='config', config_name='default.yaml')
def main(config):

    script_path = os.path.dirname(os.path.realpath(__file__))
    print(f'Script path: {script_path}')
    if not os.path.exists(os.path.join(script_path, 'outputs/eval_3duis')):
        os.makedirs(os.path.join(script_path, 'outputs/eval_3duis'))


    logging.info('===> Configurations')
    logging.info(OmegaConf.to_yaml(config))
    device = 'cpu'  # all cpu for parallelization

    # Dataloader
    logging.info('===> Initializing dataloader')
    DatasetClass = load_dataset(config.data.dataset)
    data_loader = initialize_data_loader(DatasetClass, config=config, phase=config.train.val_phase,
                                         num_workers=config.data.num_workers, augment_data=False,
                                         shuffle=False, repeat=False, batch_size=1)
    dataset = data_loader.dataset

    # Model initialization
    logging.info('===> Building 3D model')
    num_in_channel = config.net.num_in_channels
    num_labels = data_loader.dataset.NUM_LABELS
    NetClass = load_model(config.net.model)
    model_3d = NetClass(num_in_channel, num_labels, config).eval().to(device)

    # Load pretrained weights into model
    print('===> Loading weights for 3D backbone: ' + config.net.weights_for_inner_model)
    state = torch.load(config.net.weights_for_inner_model, map_location=torch.device('cpu'))
    matched_weights = load_state_with_same_shape(model_3d, state['state_dict'])
    model_dict = model_3d.state_dict()
    model_dict.update(matched_weights)
    model_3d.load_state_dict(model_dict)

    # Add instance segmentation
    evaluator = InstanceEvaluator(['foreground'], [1])

    # Start multiprocessing for evaluation
    # with tqdm_joblib(tqdm(desc="3DUIS for ScanNet val", total=len(dataset))) as progress_bar:
        # refined_instance_predictions = Parallel(n_jobs=24, backend="threading")(delayed(handle_scene)(i, dataset, model_3d, script_path, refine=True) for i in np.arange(len(dataset)))

    scene_ids = np.arange(len(dataset))
    np.random.shuffle(scene_ids)
    refined_instance_predictions = [handle_scene(i, dataset, model_3d, script_path, refine=True) for i in scene_ids]

    # Evaluate for all scenes
    for i, (pred_ins, index, target, instances) in enumerate(refined_instance_predictions):
        print(f'Evaluating scene {i + 1} of {len(refined_instance_predictions)}')

        unique_segments = np.unique(pred_ins)
        partitions = np.zeros((unique_segments.shape[0], target.shape[0]), dtype=bool)
        for segment_index, segment_id in enumerate(unique_segments):
            partitions[segment_index, pred_ins == segment_id] = True

        # Add to evaluator
        batch_predictions = {}
        for inst_pred_id in range(partitions.shape[0]):
            batch_predictions[inst_pred_id] = {'conf': 0.99 - (inst_pred_id * 10e-3),
                                               'label_id': 1,
                                               'pred_mask': partitions[inst_pred_id]}

        evaluator.add_prediction(batch_predictions, index)
        evaluator.add_gt(target * 1000 + instances, index)

    all_mAP, mAP50, mAP25, all_mAR, mAR50, mAR25 = evaluator.evaluate()
    print('mAP: ', all_mAP)
    print('mAP50: ', mAP50)
    print('mAP25: ', mAP25)


if __name__ == "__main__":
    main()

