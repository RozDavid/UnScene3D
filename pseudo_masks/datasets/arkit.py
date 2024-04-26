from lib.datasets.scannet import *

class ARKit_Dataset(ScanNet_Dataset):

    CLASS_LABELS = ['background', 'foreground']
    VALID_CLASS_IDS = [0, 1]
    INSTANCE_IGNORE_LABELS = [0]
    NUM_LABELS = 2
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 cache=False,
                 verbose=False,
                 phase: str = 'train',
                 data_root: str = None):

        if 'train' not in phase.lower():
            self.CLIP_BOUND = self.TEST_CLIP_BOUND

        self.config = config
        self.phase = phase

        data_root = config.data.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        self.data_paths = np.array(sorted(read_txt(os.path.join(data_root, f'{phase}.txt'))))
        self.scene_names = [Path(data_path).stem for data_path in self.data_paths]
        logging.info('Loading {} scenes into {} for {} phase'.format(len(self.data_paths), self.__class__.__name__, phase))

        # Init DatasetBase
        super(ScanNet_Dataset, self).__init__(
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

        # Create label map
        self.label_map = {0: 0, 1: 1, self.ignore_mask: self.ignore_mask}
        self.label_mapper = np.vectorize(lambda x: self.label_map[x])
        self.NUM_LABELS = len(self.VALID_CLASS_IDS)

        # Precompute a mapping from ids to categories
        self.id2cat_name = {}
        self.cat_name2id = {}
        for id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            self.id2cat_name[id] = cat_name
            self.cat_name2id[cat_name] = id

    def clean_mesh(self, scene_mesh):

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

        return scene_mesh

    def clean_segments(self, comps, min_vert_num=500):
        unique_comps, unique_numbers = np.unique(comps, return_counts=True)
        invalid_comp_ids = unique_comps[unique_numbers < min_vert_num]
        valid_comps = ~np.isin(comps, invalid_comp_ids)
        return comps[valid_comps], valid_comps

    def load_mesh(self, index):
        filepath = self.data_root / self.data_paths[index]
        scene_mesh = o3d.io.read_triangle_mesh(str(filepath))

        # For ArKit it can have extra vertices
        scene_mesh = self.clean_mesh(scene_mesh)

        # Parse to correct type and call algorithm
        vertices = np.array(scene_mesh.vertices)
        colors = np.array(scene_mesh.vertex_colors) * 255.
        faces = np.array(scene_mesh.triangles)

        return scene_mesh, faces, vertices, colors, np.zeros(vertices.shape[0]), np.zeros(vertices.shape[0]), filepath.stem


    def load_scene_data(self, index):

        mesh, faces, coords, feats, labels, instance_ids, scene_name = self.load_mesh(index)

        all_seg_indices, all_seg_connectivity = None, None
        if self.config.data.segments_as_grids:
            all_seg_indices, all_seg_connectivity = [], []
            seg_min_point_nums = self.config.data.segments_min_vert_nums

            valid_seg_indices = None
            for seg_min_point_num in seg_min_point_nums:
                seg_indices, seg_connectivity = felzenszwalb_cpp.segment_mesh(coords, faces, feats, 0.005, seg_min_point_num)
                all_seg_indices += [seg_indices]
                all_seg_connectivity += [seg_connectivity]

                comps, valid_comps = self.clean_segments(seg_indices, min_vert_num=seg_min_point_num)
                valid_seg_indices = valid_comps

            all_seg_indices = np.stack(all_seg_indices, axis=-1)


            # remove too small segments
            coords = coords[valid_seg_indices]
            feats = feats[valid_seg_indices]
            labels = labels[valid_seg_indices]
            instance_ids = instance_ids[valid_seg_indices]
            all_seg_indices = all_seg_indices[valid_seg_indices]

        # no rgb data here
        rgb_images, poses, color_intrinsics = None, None, None

        # push coords around the center
        coords -= coords.min(0)

        return coords, feats, labels, instance_ids, scene_name,  rgb_images, poses, color_intrinsics, all_seg_indices, all_seg_connectivity


    def __getitem__(self, index):

        # Load data
        coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity = self.load_scene_data(index)

        if not ((coords.shape[1] == 3) and (coords.shape[0] == feats.shape[0]) and (coords.shape[0] != 0)):
            print(f'Detected problem with {scene_name}')
            print(coords.shape, feats.shape)
            return self.__getitem__(0)

        # Voxelize and augment
        coords, feats, labels, instance_ids, camera_poses, segment_ids, transformations = self.prepare_scene_data(coords, feats, labels, instance_ids, camera_poses, segment_ids)

        # Collect in tuple
        return_args = (coords, feats, labels, instance_ids, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, transformations[1].astype(np.float32))

        return return_args

    def scene_name_to_index(self, scene_name):
        return self.scene_names.index(scene_name)

    def index_to_scene_name(self, index):
        return self.scene_names[index]


class ARKit_2cmDataset(ARKit_Dataset):
    VOXEL_SIZE = 0.02


class ARKit_10cmDataset(ARKit_Dataset):
    VOXEL_SIZE = 0.10

