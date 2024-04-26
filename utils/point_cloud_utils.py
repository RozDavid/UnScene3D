from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d
from plyfile import PlyData, PlyElement


def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.uint8).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels


def load_ply_with_normals(filepath):
    mesh = open3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    coords, feats, labels = load_ply(filepath)
    assert np.allclose(coords, vertices), "different coordinates"
    feats = np.hstack((feats, normals))

    return coords, feats, labels


def load_obj_with_normals(filepath):
    mesh = open3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    coords = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)
    feats = np.hstack((colors, normals))

    return coords, feats


def write_point_cloud_in_ply(
    filepath: Path,
    coords: np.ndarray,
    feats: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    dtypes: Optional[List[Tuple[str, str]]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "<u2"),
    ],
    comments: Optional[List[str]] = [""],
):
    combined_coords = tuple([coords])
    if feats is not None:
        combined_coords += tuple([feats])
    else:
        dtypes = dtypes[:3] + dtypes[-1:]
    if labels is not None:
        combined_coords += tuple([labels[:, np.newaxis]])
    else:
        dtypes = dtypes[:-1]
    combined_coords = np.hstack(combined_coords)
    ply_data = np.empty(len(coords), dtype=dtypes)
    for i, dtype in enumerate(dtypes):
        ply_data[dtype[0]] = combined_coords[:, i]
    ply_data = PlyData([PlyElement.describe(ply_data, "vertex", comments=comments)])
    ply_data.write(filepath)


def separate_segments(foreground, unique_segments, connectivity_dict):
    
    # Separate non-connected regions
    separated_instances = []  # the fused blobs
    curr_mask_segment_ids = unique_segments[foreground]

    # Iterate over all segments in query mask
    for c in curr_mask_segment_ids:

        # check if any set contains one of it's neighbours
        # if multiple contains, wer should merge those cells
        neighbour_segments = connectivity_dict[c]
        last_fused_match = -1
        merged = False

        # iterate over all past blobs, associate and potentially merge if it was a bridge segment
        fused_id = 0
        while fused_id < len(separated_instances):

            fused_segments = separated_instances[fused_id]
            if len(neighbour_segments.intersection(fused_segments)) != 0:
                merged = True
                fused_segments.add(c)
                if last_fused_match != -1:
                    # merge with the previous segment, then delete
                    separated_instances[last_fused_match] = separated_instances[last_fused_match].union(fused_segments)
                    separated_instances.pop(fused_id)
                else:
                    last_fused_match = fused_id

            fused_id += 1

        # add as new segment if never associated with others
        if not merged:
            separated_instances += [{c}]

    # Convert to list of lists
    separated_instances = [list(x) for x in separated_instances]

    return separated_instances
