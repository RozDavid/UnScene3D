import copy
import os
import sys
import numpy as np
import open3d as o3d
import time

def do_dbscan(vertices, colors, faces):
    import hdbscan

    # Do the same thing with DBScan
    start = time.time()
    dbscan_pcd = o3d.geometry.PointCloud()
    dbscan_pcd.points = o3d.utility.Vector3dVector(vertices)
    dbscan_pcd.colors = o3d.utility.Vector3dVector(colors)
    dbscan_pcd.estimate_normals()

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=500, min_samples=1)
    clusterer.fit(np.concatenate((vertices, colors, np.array(dbscan_pcd.normals)), axis=1))
    comps = clusterer.labels_

    end_dbscan = time.time()
    print(f"HDBSCAN took {end_dbscan - end_dbscan:.4f} s")

    return comps, np.zeros((0, 2))

def do_felzenszwalb(vertices, colors, faces, num_points):
    import felzenszwalb_cpp

    start = time.time()
    comps, connectivity = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, 0.005, num_points)  # orig was 50
    unique_segments = np.unique(comps)
    end = time.time()
    print(f"Felzenswalb took {end - start:.4f} s")
    return comps, connectivity


def clean_mesh(scene_mesh):

    start = time.time()

    # Remove vertices which are not connected to any face
    vertices = np.array(scene_mesh.vertices).astype(np.single)
    colors = np.array(scene_mesh.vertex_colors).astype(np.single)
    faces = np.array(scene_mesh.triangles).astype(np.intc)

    # Get the vertices those are associated with the faces
    valid_vertices = np.unique(faces)
    unreferenced_vertices = np.setdiff1d(np.arange(vertices.shape[0]), valid_vertices)

    # Mark the unreferenced vertices
    removed_vertices = np.zeros(vertices.shape[0])
    removed_vertices[unreferenced_vertices] = 1

    # Reindex cumsum removed vertices and subtract from faces array
    removed_cumsum = np.cumsum(removed_vertices)
    cumsum_faces = removed_cumsum[faces]
    updated_faces = (faces - cumsum_faces).astype(np.intc)

    # Update and return the scene mesh
    scene_mesh.vertices = o3d.utility.Vector3dVector(vertices[valid_vertices])
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[valid_vertices])
    scene_mesh.triangles = o3d.utility.Vector3iVector(updated_faces)
    scene_mesh.compute_vertex_normals()

    end = time.time()
    print(f"Cleaning took {end - start:.4f} s")

    return scene_mesh

def clean_segments(comps, min_vert_num=500):
    unique_comps, unique_numbers = np.unique(comps, return_counts=True)
    invalid_comp_ids = unique_comps[unique_numbers < min_vert_num]
    valid_comps = ~np.isin(comps, invalid_comp_ids)
    return comps[valid_comps], valid_comps


def main(min_vert_num=50):
    # Define paths
    # mesh_path = '/mnt/cluster/himring/drozenberszki/Datasets/ScanNet/scans/scene0000_00/scene0000_00_vh_clean_2.ply'
    mesh_path = '/mnt/cluster/himring/drozenberszki/Datasets/ScanNet/scans/scene0427_00/scene0427_00_vh_clean_2.ply'
    # mesh_path = '/mnt/cluster/himring/drozenberszki/Datasets/ArKitScenes/meshes/low_res/416407.ply'
    # mesh_path = '/cluster/eriador/cyeshwanth/datasets/ARKitScenes/laser_scanner_point_clouds/416411/1mm/chunks_depth9/mesh/simplified_0.0156_mesh_aligned.ply'

    # Load meshes, simplify and calculate normals
    scene_mesh = o3d.io.read_triangle_mesh(mesh_path)
    scene_mesh = scene_mesh.simplify_vertex_clustering(voxel_size=0.02, contraction=o3d.geometry.SimplificationContraction.Average)

    scene_mesh = clean_mesh(scene_mesh)

    # Parse to numpy
    vertices = np.array(scene_mesh.vertices).astype(np.single)
    colors = np.array(scene_mesh.vertex_colors).astype(np.single)
    faces = np.array(scene_mesh.triangles).astype(np.intc)
    shift = [(vertices[:, 0].max() - vertices[:, 0].min()) * 1.2, 0., 0.]

    # Clean vertices which are not connected to any face
    valid_vertices = np.unique(faces)
    vertices = vertices[valid_vertices]
    colors = colors[valid_vertices]

    # call algorithm
    comps, connectivity = do_felzenszwalb(vertices, colors, faces, num_points=min_vert_num)

    # Clean segments
    # comps, valid_comps = clean_segments(comps, min_vert_num=min_vert_num)
    unique_comps, unique_numbers = np.unique(comps, return_counts=True)

    # visualize data
    random_colors = np.stack((comps * 217 % 256, comps * 217 % 311, comps * 217 % 541)).T % 256  # random numbers for visualizaton
    scene_mesh.vertex_colors = o3d.utility.Vector3dVector(random_colors / 255.)
    o3d.visualization.draw_geometries([scene_mesh])

    # for uc in np.unique(connectivity):
    #     segment_mesh = copy.copy(scene_mesh)
    #     segment_mesh.vertices = o3d.utility.Vector3dVector(vertices + shift)
    #
    #     connected_segments = connectivity[connectivity[:, 0] == uc, 1]
    #
    #     connected_component_colors = np.zeros(random_colors.shape)
    #     connected_component_colors[comps == uc] = [0.7, 0., 0.]
    #     connected_component_colors[np.isin(comps, connected_segments)] = [0., 0.7, 0.]
    #
    #     segment_mesh.vertex_colors = o3d.utility.Vector3dVector(connected_component_colors)
    #     o3d.visualization.draw_geometries([scene_mesh, segment_mesh])


# python main function
if __name__ == '__main__':

    if len(sys.argv) == 1:
        vert_nums = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for vert_num in vert_nums:
            main(vert_num)
    else:
        vert_num = int(sys.argv[1])
        main(vert_num)
