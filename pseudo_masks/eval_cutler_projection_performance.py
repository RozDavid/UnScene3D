import hydra
import logging
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from MinkowskiEngine import SparseTensor
from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from utils.cuda_utils.raycast_image import Project2DFeaturesCUDA
from utils.freemask_utils import *
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append('./ext/CutLER/cutler')
sys.path.append('./ext/CutLER/cutler/demo')
from detectron2.config import get_cfg

from predictor import VisualizationDemo

# define hard coded paths
config_path = './ext/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml'
model_path = './ext/CutLER/models/cutler_cascade_final.pth'

def add_cutler_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    cfg.TEST.NO_SEGM = False


def setup_cfg(config_file= '', opts=[], confidence_threshold=0.35):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):

    device = 'cuda'
    visualize = False

    # Dataloader
    logging.info('===> Initializing dataloader')
    DatasetClass = load_dataset(config.data.dataset)
    data_loader = initialize_data_loader(DatasetClass, config=config, phase='val',
                                         num_workers=config.data.num_workers, augment_data=False,
                                         shuffle=False, repeat=False, batch_size=1)
    dataset = data_loader.dataset

    # Add instance segmentation
    evaluator = InstanceEvaluator(['foreground'], [1])

    # load config and model
    logging.info('===> Initializing CutLER model')
    pts = ['MODEL.WEIGHTS', model_path]
    cfg = setup_cfg(config_path,  ['MODEL.WEIGHTS', model_path])
    demo = VisualizationDemo(cfg)

    # Load model for single image projection
    demo = VisualizationDemo(cfg)

    # Define fetaure projecter
    prediction_projecter = Project2DFeaturesCUDA(width=dataset.depth_shape[1],
                                                 height=dataset.depth_shape[0],
                                                 voxel_size=dataset.VOXEL_SIZE,
                                                 config=config,
                                                 depth_max=3.)

    # create directory to save outputs
    if not os.path.exists('outputs/eval_cutler'):
        os.makedirs('outputs/eval_cutler')

    # Start main iteration over the dataset
    tqdm_loader = tqdm(data_loader)
    for batch_id, batch in enumerate(tqdm_loader):
        coords, feats, target, instances, scene_name, images, camera_poses, color_intrinsics, segment_ids, seg_connectivity, *transform = batch
        coords, feats, target, instances, camera_poses, color_intrinsics = coords.to(device), feats.to(device), target.to(device), instances.to(device), camera_poses.to(device), color_intrinsics.to(device)

        # Make target labels to freemask labels
        target[target > 1] = 1
        target[target < 1] = 0

        sinput = SparseTensor(feats, coords)
        starget = SparseTensor(target.float().view(-1, 1),
                               coordinate_manager=sinput.coordinate_manager,
                               coordinate_map_key=sinput.coordinate_map_key)

        out_fname = os.path.join(f'outputs/eval_cutler/cutler_preds_{scene_name[0]}.npy')
        out_coords_name = os.path.join(f'outputs/eval_cutler/cutler_coords_{scene_name[0]}.npy')
        if os.path.exists(out_fname):
            np.save(out_coords_name, coords[:, 1:].cpu().numpy() * dataset.VOXEL_SIZE)
            print(f"Scene already processed {scene_name[0]}")
            continue

        # do prediction and iteration for all images
        all_predictions = 0
        accumulated_preds = torch.zeros(len(sinput), device=sinput.device, dtype=torch.long)
        accumulated_confs = torch.zeros(len(sinput), device=sinput.device, dtype=torch.long)
        for img_id, s_image in enumerate(images[0]):

            # predict with image
            img_np = (s_image.permute(1, 2, 0).cpu().numpy() + 1) / 2.
            img_np = (img_np[..., ::-1] * 255.).astype(np.uint8)
            predictions, visualized_output = demo.run_on_image(img_np)

            # visualize for debugging
            if visualize:
                plt.imshow(visualized_output.get_image()[:, :, ::-1])
                plt.show()

            # parse redictions and confidences
            scores = []
            masks = []
            num_preds, img_shape = len(predictions['instances']), img_np.shape[:2]
            for pred_id in range(num_preds):
                score = predictions['instances'][pred_id].scores
                mask = predictions['instances'][pred_id].pred_masks.squeeze()

                score_mask = torch.ones(mask.shape, device=mask.device) * mask * score
                scores += [score_mask]
                masks += [mask.long() * pred_id]

            if len(scores) == 0:
                scores = [torch.zeros(img_shape, device=device)]

            scores = torch.stack(scores)
            max_scores, pred_ids = scores.max(dim=0)
            pred_ids += all_predictions

            # mask scores should be int for confidence to do prediction
            max_scores = (max_scores * 100).int()

            # project both the confidence and the prediction
            curr_projected_preds, mapping2dto3d_num = prediction_projecter(encoded_2d_features=pred_ids.view(1, 1, *pred_ids.shape, 1).int().contiguous(),
                                                                           coords=sinput.C,
                                                                           view_matrix=camera_poses[:, img_id: img_id + 1, ...],
                                                                           intrinsic_params=color_intrinsics.to(device),
                                                                           pred_mode=True)

            # for confidences, we take 1% accuracy
            curr_projected_confidences, mapping2dto3d_num = prediction_projecter(encoded_2d_features=max_scores.view(1, 1, *max_scores.shape, 1).int().contiguous(),
                                                                                 coords=sinput.C,
                                                                                 view_matrix=camera_poses[:, img_id: img_id + 1, ...],
                                                                                 intrinsic_params=color_intrinsics.to(device),
                                                                                 pred_mode=True)

            # update more confident predictions
            update_mask = curr_projected_confidences > accumulated_confs
            accumulated_preds[update_mask] = curr_projected_preds[update_mask]
            accumulated_confs[update_mask] = curr_projected_confidences[update_mask]
            all_predictions += num_preds

        # Format to predictions compatible with evaluator
        accumulated_preds = accumulated_preds.cpu().numpy()
        unique_segments = np.unique(accumulated_preds)
        partitions = np.zeros((unique_segments.shape[0], accumulated_preds.shape[0]), dtype=bool)
        for i, segment_id in enumerate(unique_segments):
            partitions[i, accumulated_preds == segment_id] = True

        # Add to evaluator
        batch_predictions = {}
        for inst_pred_id in range(partitions.shape[0]):
            batch_predictions[inst_pred_id] = {'conf': 0.99 - (inst_pred_id * 10e-3),
                                               'label_id': 1,
                                               'pred_mask': partitions[inst_pred_id]}

        evaluator.add_prediction(batch_predictions, batch_id)
        evaluator.add_gt((target * 1000 + instances).cpu().numpy(), batch_id)

        # Save predictions
        np.save(out_fname, accumulated_preds)

    # Finally, evaluate the full dataset
    all_mAP, mAP50, mAP25, all_mAR, mAR50, mAR25 = evaluator.evaluate()
    print('mAP: ', all_mAP)
    print('mAP50: ', mAP50)
    print('mAP25: ', mAP25)


if __name__ == "__main__":
    main()

