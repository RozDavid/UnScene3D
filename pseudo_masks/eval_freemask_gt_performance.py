import hydra
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from models import load_model
from unscene3d_pseudo_main import segment_scene
from models.encoders_2d import load_2d_model
from utils.utils import load_state_with_same_shape
from utils.freemask_utils import *
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
import pyviz3d.visualizer as vis

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path='config', config_name='default.yaml')
def main(config):

    logging.info('===> Configurations')
    logging.info(OmegaConf.to_yaml(config))
    device = 'cuda'
    visualize = False

    # Init WanDB project
    # start a new wandb run to track this script
    wandb_cfg = OmegaConf.to_container(
        config, resolve=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="maskcut_experiments",
        name=f'masked_ncut_{config.freemask.modality}_{config.freemask.affinity_tau}_{config.data.segments_min_vert_nums[0]}',
        config=wandb_cfg,
    )

    # Dataloader
    logging.info('===> Initializing dataloader')
    DatasetClass = load_dataset(config.data.dataset)
    data_loader = initialize_data_loader(DatasetClass, config=config, phase='val',
                                         num_workers=1, augment_data=False,
                                         shuffle=False, repeat=False, batch_size=1)
    dataset = data_loader.dataset  # type: lib.datasets.scannet_free.ScanNetFree_2cmDataset

    # Add instance segmentation
    evaluator = InstanceEvaluator(['foreground'], [1])

    # Model initialization
    logging.info('===> Building 3D model')
    num_in_channel = config.net.num_in_channels
    num_labels = data_loader.dataset.NUM_LABELS
    NetClass = load_model(config.net.model)
    model_3d = NetClass(num_in_channel, num_labels, config)
    model_3d = model_3d.eval().to(device)

    # Load pretrained weights into model
    print('===> Loading weights for 3D backbone: ' + config.net.weights_for_inner_model)
    state = torch.load(config.net.weights_for_inner_model)
    matched_weights = load_state_with_same_shape(model_3d, state['state_dict'])
    model_dict = model_3d.state_dict()
    model_dict.update(matched_weights)
    model_3d.load_state_dict(model_dict)

    # Pick which modality to use
    if config.freemask.modality == 'color':
        # Load models
        logging.info('===> Building 2D model')
        ImageEncoderClass = load_2d_model(config.image_data.model)
        model_2d = ImageEncoderClass(config, data_loader.dataset).eval().to(device)

        model = model_2d
    elif config.freemask.modality == 'geom':
        model = model_3d
        config.image_data.use_images = False
    elif config.freemask.modality == 'both':

        # Load models
        logging.info('===> Building 2D model')
        ImageEncoderClass = load_2d_model(config.image_data.model)
        model_2d = ImageEncoderClass(config, data_loader.dataset).eval().to(device)

        model = (model_2d, model_3d)
    else:
        raise ValueError('Unknown modality')

    # Start main iteration over the dataset
    tqdm_loader = tqdm(data_loader)
    for batch_id, batch in enumerate(tqdm_loader):
        coords, feats, target, instances, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, *transform = batch
        coords, feats, target, instances, segment_ids, seg_connectivity = coords.to(device), feats.to(device), target.to(device), instances.to(device), segment_ids[:, config.freemask.segment_dimension_order].to(device), torch.from_numpy(
            seg_connectivity[0][config.freemask.segment_dimension_order]).long().to(device)
        tqdm_loader.set_description(scene_name[0])

        # Make target labels to freemask labels
        target[target > 1] = 1
        target[target < 1] = 0

        # Create pyviz3d visualizer and add original point cloud
        v = vis.Visualizer()
        orig_coords = coords[:, 1:].cpu().numpy() * dataset.VOXEL_SIZE
        v.add_points("RGB", orig_coords,
                     colors=(feats.cpu().numpy() + 0.5) * 255.,
                     visible=True,
                     point_size=50)

        # Add instance segmentation with random colors
        instance_colors = np.random.randint(0, 255, size=(instances.max().item() + 1, 3))
        instance_colors[0] = [0, 0, 0]
        v.add_points("instseg", orig_coords,
                        colors=instance_colors[instances.cpu().numpy()],
                        visible=True,
                        point_size=50)

        # Add semantic segmentation with random colors
        semantic_colors = np.random.randint(0, 255, size=(target.max().item() + 1, 3))
        semantic_colors[0] = [0, 0, 0]
        v.add_points("semseg", orig_coords,
                        colors=semantic_colors[target.cpu().numpy()],
                        visible=True,
                        point_size=50)

        # Add segment ids with random colors
        segment_colors = np.random.randint(0, 255, size=(segment_ids.max().item() + 1, 3))
        v.add_points("segments", orig_coords,
                        colors=segment_colors[segment_ids.cpu().numpy()],
                        visible=True,
                        point_size=50)
        # v.save(f'outputs/{scene_name[0]}')

        # Visualizations
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
        orig_pcd.colors = o3d.utility.Vector3dVector(feats.cpu().numpy() + 0.5)
        shift = [(coords[:, 1].max() - coords[:, 1].min()).cpu().numpy() * 1.2, 0., 0.]

        # Calculate inverse segment mapping
        unique_segments = segment_ids.unique()
        inverse_segment_mapping = torch.zeros_like(segment_ids)
        for i, s_id in enumerate(unique_segments):
            segment_mask = segment_ids == s_id
            inverse_segment_mapping[segment_mask] = i

        # Get MaksCut3D predictions
        if config.freemask.mode == 'maskcut3d':
            partitions, _ = segment_scene(coords, feats, batch, model, config, dataset, segment_ids, seg_connectivity, visualize=False, filter_large_segments=True)
        elif config.freemask.mode == 'freemask':
            pass
        elif config.freemask.mode == 'segment':
            unique_segments = segment_ids.unique()
            partitions = torch.zeros((unique_segments.shape[0], segment_ids.shape[0]), dtype=torch.bool, device=segment_ids.device)
            for i, segment_id in enumerate(unique_segments):
                partitions[i, segment_ids == segment_id] = True
            partitions = partitions.cpu().numpy()


        elif config.freemask.mode == 'dbscan':
            dbscan_pcd = o3d.geometry.PointCloud()
            dbscan_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy() * dataset.VOXEL_SIZE)
            dbscan_pcd.colors = o3d.utility.Vector3dVector(feats.cpu().numpy() + 0.5)
            dbscan_pcd.estimate_normals()

            dbscan_input = np.concatenate((np.array(dbscan_pcd.points), np.array(dbscan_pcd.colors), np.array(dbscan_pcd.normals)), axis=1)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=config.data.segments_min_vert_nums[0], min_samples=1).fit(dbscan_input)
            clusters = clusterer.labels_ + 1

            unique_segments = np.unique(clusters)
            partitions = np.zeros((unique_segments.shape[0], segment_ids.shape[0]), dtype=bool)
            for i, segment_id in enumerate(unique_segments):
                partitions[i, clusters == segment_id] = True
        else:
            raise ValueError('Unknown mode')


        # Add to evaluator
        batch_predictions = {}
        for inst_pred_id in range(partitions.shape[0]):
            batch_predictions[inst_pred_id] = {'conf': 0.99 - (inst_pred_id * 10e-3),
                                               'label_id': 1,
                                               'pred_mask': partitions[inst_pred_id]}

        evaluator.add_prediction(batch_predictions, batch_id)
        evaluator.add_gt((target * 1000 + instances).cpu().numpy(), batch_id)

        # Visualize the final bipartitions
        if visualize:
            instance_colors = np.zeros(coords[:, 1:].shape)
            for i, bipartition in enumerate(partitions):
                instance_colors[bipartition] = np.random.rand(3)
            instance_pcd = o3d.geometry.PointCloud()
            instance_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy() + shift)
            instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)

            o3d.visualization.draw_geometries([instance_pcd, orig_pcd])

    all_mAP, mAP50, mAP25, all_mAR, mAR50, mAR25 = evaluator.evaluate()
    print(f"Results for experiments using image data {config.image_data.use_images} and {config.freemask.mode} freemask mode and affinity Tau {config.freemask.affinity_tau} and min vert num {config.data.segments_min_vert_nums[0]}")
    print('mAP: ', all_mAP)
    print('mAP50: ', mAP50)
    print('mAP25: ', mAP25)

if __name__ == "__main__":
    main()

