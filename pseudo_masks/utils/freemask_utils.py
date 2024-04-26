import numpy as np
import open3d as o3d
import hdbscan

import torch
from MinkowskiEngine import SparseTensor

def cosine_sim(feats_k, feats_q):

    # Normalize both keys and queries, calculate attention and normalize attention too
    eps = 10e-10
    key_feats = feats_k / (feats_k.norm(dim=1, keepdim=True) + eps)
    queries = feats_q / (feats_q.norm(dim=1, keepdim=True) + eps)
    attn = queries @ key_feats.T
    attn -= attn.min(-1, keepdim=True)[0]
    attn /= attn.max(-1, keepdim=True)[0] + eps

    return attn

def l2_sim(feats_k, feats_q):

    # subtract every query from every key
    query_num = feats_q.shape[0]
    key_num = feats_k.shape[0]
    feature_dim = feats_k.shape[-1]

    # sadly we need the iteration for memory reasons
    attn = torch.zeros((query_num, key_num), device=feats_k.device)
    for k_id in range(key_num):
        attn[:, k_id] = torch.linalg.norm((feats_q - feats_k[k_id]), dim=-1)

    attn -= attn.min(-1, keepdim=True)[0]
    attn /= attn.max(-1, keepdim=True)[0]

    # similarity is high close
    return 1. - attn

def lidar_3duis(model, coords, features, device='cuda', z_max = 0.25):

    # Remove ground labels
    # Calculate normals of coords and get clusters
    vertices, colors = coords[:, 1:].cpu().numpy(), features.cpu().numpy() + features.min().item()
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
    floor_outliers = np.ones(len(vertices), dtype=np.bool)
    floor_outliers[floor_inliers] = False

    # HDBSCAN clustering and add floor as cluster 1
    clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=1)
    clusterer.fit(np.concatenate((vertices, colors, np.array(dbscan_pcd.normals)), axis=1))
    comps = clusterer.labels_ + 2
    comps[floor_inliers] = 1

    # visualize data
    random_colors = np.stack((comps * 217 % 256, comps * 217 % 311, comps * 217 % 541)).T % 256  # random numbers for visualizaton
    dbscan_pcd.colors = o3d.utility.Vector3dVector(random_colors / 255.)
    o3d.visualization.draw_geometries([dbscan_pcd])

    clusters_full_res = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.long)
    sinput = SparseTensor(features.to(device), coords.to(device))

    # Encode features
    _, feature_maps = model(sinput)
    out = feature_maps[f'res_1']




    # Get number of points per cluster
    ins, num_pts = np.unique(comps, return_counts=True)

