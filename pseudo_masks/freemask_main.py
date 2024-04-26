import os
import hydra
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.decomposition import PCA
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from models import load_model
from models.encoders_2d import load_2d_model
from utils.cuda_utils.raycast_image import Project2DFeaturesCUDA
from utils.pc_utils import matrix_nms
from utils.utils import load_state_with_same_shape
from constants.scannet_constants import SCANNET_COLOR_MAP_200
from utils.freemask_utils import *

from MinkowskiEngine import SparseTensor
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling, MinkowskiMaxPooling
import pytorch3d.ops.sample_farthest_points as sample_farthest_points

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):

    logging.info('===> Configurations')
    logging.info(OmegaConf.to_yaml(config))
    device = 'cuda'
    num_downscale_steps = int(np.log2(config.freemask.resolution_scale))

    # Dataloader
    logging.info('===> Initializing dataloader')
    DatasetClass = load_dataset(config.data.dataset)
    data_loader = initialize_data_loader(DatasetClass, config=config, phase='trainval',
                                         num_workers=config.data.num_workers, augment_data=False,
                                         shuffle=False, repeat=False, batch_size=1)
    dataset = data_loader.dataset  # type: lib.datasets.scannet.ScanNetDataset

    # Model initialization
    logging.info('===> Building 3D model')
    num_in_channel = config.net.num_in_channels
    num_labels = data_loader.dataset.NUM_LABELS
    NetClass = load_model(config.net.model)
    model_3d = NetClass(num_in_channel, num_labels, config)
    model_3d = model_3d.eval().to(device)

    if config.image_data.use_images:
        logging.info('===> Building 2D model')
        ImageEncoderClass = load_2d_model(config.image_data.model)
        image_model = ImageEncoderClass(config, data_loader.dataset).eval().to(device)

        feature_projecter = Project2DFeaturesCUDA(width=dataset.depth_shape[1],
                                                  height=dataset.depth_shape[0],
                                                  voxel_size=dataset.VOXEL_SIZE,
                                                  config=config)

    # Load pretrained weights into model
    if os.path.isfile(config.net.weights_for_inner_model):
        print('===> Loading weights for 3D backbone: ' + config.net.weights_for_inner_model)
        state = torch.load(config.net.weights_for_inner_model)
        matched_weights = load_state_with_same_shape(model_3d, state['state_dict'])
        model_dict = model_3d.state_dict()
        model_dict.update(matched_weights)
        model_3d.load_state_dict(model_dict)

    # Start main iteration over the dataset
    tqdm_loader = tqdm(data_loader)
    for batch_id, batch in enumerate(tqdm_loader):
        coords, feats, target, instances, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, *transform = batch
        coords, feats, target, instances= coords.to(device), feats.to(device), target.to(device), instances.to(device)

        # Skip if already processed
        if os.path.exists(f'outputs/{scene_name[0]}_cloud.npy'):
            print(f'Scene already processed: {scene_name[0]}')
            continue

        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy())
        orig_pcd.colors = o3d.utility.Vector3dVector(feats.cpu().numpy() + 0.5)
        shift = [(coords[:, 1].max() - coords[:, 1].min()).cpu().numpy() * 1.2, 0., 0.]

        sinput = SparseTensor(feats, coords)
        starget = SparseTensor(target.float().view(-1, 1),
                               coordinate_manager=sinput.coordinate_manager,
                               coordinate_map_key=sinput.coordinate_map_key)
        tqdm_loader.set_description(scene_name[0])

        # get low res targets
        lr_target = starget
        pooling_operation = MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        pooled = starget
        for s in range(num_downscale_steps):
            pooled = pooling_operation(pooled)

        # Feed forward on pretrained model
        _, feature_maps = model_3d(sinput)
        keys = feature_maps[f'res_{config.freemask.resolution_scale}']
        lr_coords = keys.C[:, 1:].cpu().numpy()

        # push images to gpu and project features to
        if images is not None and images != []:
            images, camera_poses, color_intrinsics = images.to(device), camera_poses.to(device), color_intrinsics.to(device)
            scene_image_feats = torch.zeros((sinput.C.shape[0], image_model.feature_dim), device=device)

            for i, img in enumerate(images[0]):

                features_2d, *_ = image_model(img.unsqueeze(0).unsqueeze(0))  # (batch, img_num, c, h, w)
                projected_features, mapping2dto3d_num = feature_projecter(encoded_2d_features=features_2d,
                                                                          coords=sinput.C,
                                                                          view_matrix=camera_poses[0, i].unsqueeze(0).unsqueeze(0),
                                                                          intrinsic_params=color_intrinsics)
                projected_points = mapping2dto3d_num > 0
                scene_image_feats[projected_points] = torch.stack((scene_image_feats[projected_points], projected_features[projected_points])).max(0)[0]

                # Visualize feature similarities for images
                # image_feats = features_2d.view(-1, features_2d.shape[-1]).detach()
                # features_shape = features_2d.squeeze().shape
                # for i in range(1):
                #
                #     pixel_id = np.random.choice(features_shape[0] * features_shape[1])
                #
                #     fig, ax = plt.subplots(1)
                #     similarities = cosine_sim(image_feats, image_feats[pixel_id].view(1, -1))
                #     similarities = similarities.view(features_shape[0], features_shape[1]).cpu().numpy()
                #
                #     circ = Circle((pixel_id % features_shape[1], pixel_id // features_shape[0]), 10, color=(1., 0., 0.))
                #
                #     ax.imshow(similarities)
                #     ax.add_patch(circ)
                #     plt.show()

                # image_shape = features_2d.shape[2:]
                # pca = PCA(n_components=3)
                # projected_backbone_feats = pca.fit_transform(features_2d.detach().cpu().numpy().reshape(-1, image_shape[-1]))
                # projected_backbone_feats = projected_backbone_feats - projected_backbone_feats.min()
                # projected_backbone_feats = projected_backbone_feats / projected_backbone_feats.max()
                # projected_backbone_feats = projected_backbone_feats.reshape(image_shape[0], image_shape[1], 3)
                #
                # projected_backbone_feats = np.concatenate((resize(img.permute(1, 2, 0).cpu().numpy().astype(projected_backbone_feats.dtype), image_shape[:2]) + 0.5,
                #                                            projected_backbone_feats), axis=1)
                # plt.imshow(projected_backbone_feats)
                # plt.show()

            # visualize image features with PCA
            # pca = PCA(n_components=3)  # for rgb
            # projected_feats_3d = pca.fit_transform(scene_image_feats.cpu().numpy())
            # projected_feats_3d = projected_feats_3d - projected_feats_3d.min()
            # projected_feats_3d = projected_feats_3d / projected_feats_3d.max()
            #
            # pca_2d = o3d.geometry.PointCloud()
            # pca_2d.points = o3d.utility.Vector3dVector(coords[:, 1:].cpu().numpy() - shift)
            # pca_2d.colors = o3d.utility.Vector3dVector(projected_feats_3d)

            # Save point cloud with open3d
            # o3d.io.write_point_cloud(f'/home/drozenberszki/dev/tmp/FreeSegmentation3D/tmp/pca_2d_{scene_name[0]}.ply', pca_2d)

            # o3d.visualization.draw_geometries([orig_pcd, pca_2d])

            keys = SparseTensor(scene_image_feats,
                                coordinate_manager=sinput.coordinate_manager,
                                coordinate_map_key=sinput.coordinate_map_key)
            pooling_operation = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
            for _ in range(num_downscale_steps):
                keys = pooling_operation(keys)
            lr_coords = keys.C[:, 1:].cpu().numpy()


        if segment_ids == []:

            # Get queries by downsampling/pooling
            queries_list = []
            pooling_operation = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
            curr_queries = keys
            for _ in range(num_downscale_steps):
                curr_queries = pooling_operation(curr_queries)
            queries_list.append(curr_queries.F.clone())

            # Keep only features
            queries = torch.cat(queries_list).detach()
            key_feats = keys.F.detach()
        else:

            # Get segment ids at lower resolution
            segment_dimension_order = 0
            segment_ids, seg_connectivity = segment_ids[:, segment_dimension_order].to(device), torch.from_numpy(seg_connectivity[0][segment_dimension_order]).long().cuda()
            s_segment_ids = SparseTensor(segment_ids.float().view(-1, 1).cuda(),
                                         coordinate_manager=sinput.coordinate_manager,
                                         coordinate_map_key=sinput.coordinate_map_key)

            # get low res targets
            pooling_operation = MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
            pooled = s_segment_ids
            for s in range(num_downscale_steps):
                pooled = pooling_operation(pooled)

            matching_segment_ids = pooled.F.long().flatten().clone()

            # get unique IDs at low res
            unique_segments = matching_segment_ids.unique()

            segments_pcd = o3d.geometry.PointCloud()
            segments_pcd.points = o3d.utility.Vector3dVector(lr_coords - shift)
            segment_colors = np.zeros(lr_coords.shape)
            for us in unique_segments:
                segment_colors[(matching_segment_ids == us).cpu().numpy()] = np.random.rand(3)
            segments_pcd.colors = o3d.utility.Vector3dVector(segment_colors)
            o3d.visualization.draw_geometries([orig_pcd, segments_pcd])

            # Get queries by averaging over the segment point feats
            segment_feats = torch.zeros((len(unique_segments), keys.F.shape[1]))

            # Only average valid features for every segments
            valid_mask = torch.any(keys.F != 0, dim=-1)
            for i, s_id in enumerate(unique_segments):
                segment_mask = valid_mask * (matching_segment_ids == s_id)
                if segment_mask.sum() > 0:
                    segment_feats[i, :] = keys.F[valid_mask * (matching_segment_ids == s_id)].mean(0)

            # Filter out the non-matched segments
            valid_segments = torch.any(segment_feats != 0, dim=-1)
            segment_feats = segment_feats[valid_segments]
            unique_segments = unique_segments[valid_segments]

            key_feats = segment_feats.clone().detach()
            queries = segment_feats.clone().detach()

            # precompute connectivity dict
            connectivity_dict = {}
            for s_id in unique_segments:
                connectivity_dict[s_id.item()] = set((seg_connectivity[seg_connectivity[:, 0] == s_id, 1].cpu().numpy()))

        # Use FPS sampled queries only if requested
        if config.freemask.use_fps_sampling:
            sampled_feats, sampled_ids = sample_farthest_points(queries.unsqueeze(0), K=config.freemask.fps_num_samples)
            sampled_feats, sampled_ids = sampled_feats.squeeze(), sampled_ids.squeeze()

            queries = sampled_feats

        soft_masks = cosine_sim(key_feats, queries) if config.freemask.similarity_metric == 'cos' else l2_sim(key_feats, queries)

        # Visualize attentions
        # for t in low_res_target.unique():
        #
        #     # Sample random point
        #     indexer = torch.arange(low_res_target.shape[0], device=device)
        #     low_res_target = lr_target.F.long().flatten().clone()
        #     point_ind = np.random.choice(indexer[low_res_target == t].cpu().numpy())
        #     sample_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
        #     sample_mesh = sample_mesh.translate(lr_coords[point_ind])
        #     sample_mesh.paint_uniform_color([1, 0.706, 0])
        #
        #     attn_colors = attn[point_ind].clone().detach()   # as the first n queries are the keys
        #     attn_colors = torch.cat(((1. - attn_colors).view(-1, 1), attn_colors.view(-1, 1), torch.zeros((attn_colors.shape[0], 1), device=device)), dim=1)
        #
        #     attn_pcd = o3d.geometry.PointCloud()
        #     attn_pcd.points = o3d.utility.Vector3dVector(lr_coords + shift)
        #     attn_pcd.colors = o3d.utility.Vector3dVector(attn_colors.cpu().numpy())
        #
        #     o3d.visualization.draw_geometries([orig_pcd, pcd_3d, attn_pcd, sample_mesh])


        # Filter out the zero features (probably from the image projection)
        soft_masks[:, torch.all(key_feats == 0, dim=-1)] = 0.
        masks = soft_masks >= config.freemask.hard_mask_threshold
        sum_masks = masks.sum(1)

        # Keep only valid masks where at least something is accepted
        keep = sum_masks > 10 if segment_ids == [] else sum_masks > 2  # minimum point number for a mask
        if keep.sum() == 0:
            print(f'No instances detected in scene {scene_name[0]}')
            continue

        print(f'We have {keep.sum()} candidates after attention scores filter')
        masks = masks[keep]
        soft_masks = soft_masks[keep]
        sum_masks = sum_masks[keep]

        # Separate Non connected instances which were activated with same attention
        if segment_ids != []:

            # Calculate hard thresholds
            masks = (soft_masks >= config.freemask.hard_mask_threshold).bool()

            # separate Non connected instances which were activated with same attention
            all_fused_instances = []
            all_fused_instance_num = 0
            for query_id, m in enumerate(masks):
                curr_mask_segment_ids = unique_segments[m].cpu().numpy()

                curr_instances = []  # the fused blobs

                # Iterate over all segments in query mask
                for c in curr_mask_segment_ids:

                    # check if any set contains one of it's neighbours
                    # if multiple contains, wer should merge those cells
                    neighbour_segments = connectivity_dict[c.item()]
                    last_fused_match = -1
                    merged = False

                    # iterate over all past blobs, associate and potentially merge if it was a bridge segment
                    fused_id = 0
                    while fused_id < len(curr_instances):

                        fused_segments = curr_instances[fused_id]
                        if len(neighbour_segments.intersection(fused_segments)) != 0:
                            merged = True
                            fused_segments.add(c)
                            if last_fused_match != -1:
                                # merge with the previous segment, then delete
                                curr_instances[last_fused_match] = curr_instances[last_fused_match].union(fused_segments)
                                curr_instances.pop(fused_id)
                                all_fused_instance_num -= 1
                            else:
                                last_fused_match = fused_id

                        fused_id += 1

                    # add as new segment if never associated with others
                    if not merged:
                        curr_instances += [set([c])]
                        all_fused_instance_num += 1

                all_fused_instances += [curr_instances]  # append to list to have an either empty or populated segment list which we can separate after

            # Separate standalone segments to multiple queries
            # First initialize with zeros, then copy value to all segments within fused instances
            separated_id = 0
            separated_soft_masks = torch.zeros((all_fused_instance_num, soft_masks.shape[1]), device=soft_masks.device)
            for query_id, serparated_query in enumerate(all_fused_instances):
                for separated_segment in serparated_query:
                    for segment_id_in_inst in separated_segment:
                        segment_location = torch.nonzero(unique_segments == segment_id_in_inst)[0][0]
                        separated_soft_masks[separated_id, segment_location] = soft_masks[query_id, segment_location]

                    separated_id += 1
            soft_masks = separated_soft_masks
            masks = (soft_masks >= config.freemask.hard_mask_threshold).bool()
            sum_masks = masks.sum(1)

            # Again, keep only valid masks where at least 2 or more segments are accepted
            keep = sum_masks > 3
            masks = masks[keep]
            soft_masks = soft_masks[keep]
            sum_masks = sum_masks[keep]

            print(f'We have {soft_masks.shape[0]} candidates after separation')

        # Sort by maskness and non-maxima suppression on masks
        maskness = (soft_masks * masks.float()).sum(1) / sum_masks
        sort_inds = torch.argsort(maskness, descending=True)
        maskness = maskness[sort_inds]
        soft_masks = soft_masks[sort_inds]

        # Remap segments to coords
        if segment_ids != []:

            mapped_soft_masks = torch.zeros((soft_masks.shape[0], lr_coords.shape[0]), device=soft_masks.device)
            mapped_key_feats = torch.zeros((lr_coords.shape[0], key_feats.shape[1]))
            for i, s_id in enumerate(unique_segments):
                segment_mask = matching_segment_ids == s_id
                mapped_soft_masks[:, segment_mask] = soft_masks[:, i].view(-1, 1)
                mapped_key_feats[segment_mask] = key_feats[i]

            soft_masks = mapped_soft_masks.clone()
            key_feats = mapped_key_feats.clone()
            masks = (soft_masks >= config.freemask.hard_mask_threshold).bool()
        else:
            masks = (soft_masks >= config.freemask.hard_mask_threshold).bool()

        # Also remove the attention masks where the extent is larger than given threshold
        filtered_masks = []
        scene_extents = (coords[:, 1:].max(0)[0] - coords[:, 1:].min(0)[0]).cpu().numpy()
        for mask_id in range(len(masks)):
            if torch.all(~masks[mask_id]):
                continue
            inst_coords = lr_coords[masks[mask_id].cpu().numpy()]
            inst_extent = inst_coords.max(0) - inst_coords.min(0)

            # Skip if extent is too large
            extent_ratios = inst_extent / scene_extents
            if np.any(extent_ratios[:2] > config.freemask.instance_to_scene_max_ratio):  # we only care about XY extent
                continue
            else:
                filtered_masks += [mask_id]

        # Only filter out everything if we have at least one valid mask
        if filtered_masks != []:
            masks = masks[filtered_masks]
            soft_masks = soft_masks[filtered_masks]
            maskness = maskness[filtered_masks]
            sum_masks = masks.sum(1)

        # Apply non-maxima suppression
        maskness = matrix_nms(maskness * 0, masks, sum_masks, maskness, sigma=2, kernel='mask')

        sort_inds = torch.argsort(maskness, descending=True)
        if len(sort_inds) > config.freemask.max_instance_num:
            sort_inds = sort_inds[:config.freemask.max_instance_num]
        maskness = maskness[sort_inds].cpu()
        soft_masks = soft_masks[sort_inds].cpu()

        # plt.plot(maskness)
        # plt.axvline(x=config.freemask.max_instance_num, color='b')
        # plt.axhline(y=config.freemask.nms_maskness_threshold, color='r')
        # plt.show()

        # Filter low NMS scores
        keep = maskness > config.freemask.nms_maskness_threshold
        if keep.sum() == 0:
            print(f'No candidates left after NMS. Min score was: {maskness.min()} at a thr of: {config.freemask.nms_maskness_threshold}')
            continue
        print(f'We have {keep.sum()} candidates after NMS maskness threshold filter')
        soft_masks = soft_masks[keep]

        # Create output dir for images
        if not os.path.exists(f'outputs/images/{scene_name[0]}'):
            os.makedirs(f'outputs/images/{scene_name[0]}')

        # Iterate over all soft masks for visualization
        np_coords = np.array(lr_coords)
        mask_colors = np.zeros(np_coords.shape)
        for mask_id, s_mask in enumerate(soft_masks): # enumerate(soft_masks[:5]):
            mask_colors[s_mask > config.freemask.hard_mask_threshold, :] = np.random.rand(3)

        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(np_coords + shift)
        mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

        # Visualize PCA feats
        pca = PCA(n_components=3)  # for rgb
        projected_feats_3d = pca.fit_transform(key_feats.cpu().numpy())
        projected_feats_3d = projected_feats_3d - projected_feats_3d.min()
        projected_feats_3d = projected_feats_3d / projected_feats_3d.max()

        pca_3d = o3d.geometry.PointCloud()
        pca_3d.points = o3d.utility.Vector3dVector(lr_coords - shift)
        pca_3d.colors = o3d.utility.Vector3dVector(projected_feats_3d)

        full_cloud = orig_pcd + mask_pcd + pca_3d
        o3d.io.write_point_cloud(f'segment_cloud_{scene_name[0]}.ply', full_cloud)

            # o3d.visualization.draw_geometries([orig_pcd, mask_pcd, pca_3d])

            # vis = o3d.visualization.Visualizer()
            # vis.create_window(visible=True)
            # vis.add_geometry(orig_pcd)
            # vis.add_geometry(mask_pcd)
            # vis.update_geometry(mask_pcd)
            # vis.poll_events()
            # vis.update_renderer()
            # vis.capture_screen_image(f'outputs/images/{scene_name[0]}/{scene_name[0]}_{mask_id}.jpg')
            # vis.destroy_window()

        # Iterate over all hard masks for visualization
        save_for_visual = False  # if we would like to show in a notebook or save for dataset
        if save_for_visual:

            # Visualize PCA feats
            pca = PCA(n_components=3)  # for rgb
            projected_feats_3d = pca.fit_transform(key_feats.cpu().numpy())
            projected_feats_3d = projected_feats_3d - projected_feats_3d.min()
            projected_feats_3d = projected_feats_3d / projected_feats_3d.max()

            pca_3d = o3d.geometry.PointCloud()
            pca_3d.points = o3d.utility.Vector3dVector(lr_coords - shift)
            pca_3d.colors = o3d.utility.Vector3dVector(projected_feats_3d)

            # Visualize all masks
            mask_pcds = []
            low_res_coords = keys.C[:, 1:].cpu().numpy()
            lr_point_num = low_res_coords.shape[0]
            for i, m in enumerate(masks.cpu().numpy()):
                inst_colors = np.zeros((lr_point_num, 3))
                inst_colors[m] = np.array(list(SCANNET_COLOR_MAP_200.values())[i % 200]) / 255.

                inst_pcd = o3d.geometry.PointCloud()
                inst_pcd.points = o3d.utility.Vector3dVector(low_res_coords + shift)
                inst_pcd.colors = o3d.utility.Vector3dVector(inst_colors)
                mask_pcds += [inst_pcd]
                # o3d.io.write_point_cloud(f'outputs/{scene_name[0]}/mask_{i}.ply', inst_pcd)
                # o3d.visualization.draw_geometries([orig_pcd, inst_pcd])

            np.save(f'outputs/{scene_name[0]}/all_masks.npy', soft_masks.detach().cpu().numpy())
            np.save(f'outputs/{scene_name[0]}/full_cloud.npy', np.concatenate((np.array(orig_pcd.points), np.array(orig_pcd.colors) * 255), axis=1))
            np.save(f'outputs/{scene_name[0]}/pca_cloud.npy', np.concatenate((np.array(pca_3d.points), np.array(pca_3d.colors) * 255), axis=1))
            np.save(f'outputs/{scene_name[0]}/feature_cloud.npy', np.concatenate((np.array(pca_3d.points), key_feats.cpu().numpy()), axis=1))
            o3d.io.write_point_cloud(f'outputs/{scene_name[0]}/full_cloud_and_pca_{batch_id}.ply', orig_pcd + pca_3d)
        else:

            # Map soft masks back to full cloud
            all_masks = soft_masks.detach().cpu().numpy().T

            # Get full res scene data to save for dataset
            full_res_scene_data = dataset.get_full_cloud_by_scene_name(scene_name[0])
            full_res_coords, full_res_feats, full_res_labels, full_res_instance_ids, *_ = full_res_scene_data

            # Calculate associations to make the visualization more dense
            small_res_tree = KDTree(lr_coords + 2)  # for rounding shift at 1/4 resolution the rounding shift was 2
            _, lr_hr_matches = small_res_tree.query(full_res_coords / dataset.VOXEL_SIZE, k=1)

            all_masks_save = all_masks[lr_hr_matches]
            full_res_segment_ids = segment_ids[lr_hr_matches].cpu().numpy() if segment_ids != [] else np.zeros_like(full_res_labels)
            np.save(f'outputs/{scene_name[0]}_cloud.npy', np.concatenate((full_res_coords, full_res_feats, full_res_labels[:, None], full_res_instance_ids[:, None], full_res_segment_ids[:, None]), axis=1))
            np.save(f'outputs/{scene_name[0]}_masks.npy', all_masks_save)


if __name__ == '__main__':
    main()


