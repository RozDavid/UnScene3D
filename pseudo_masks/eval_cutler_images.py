import hydra
import logging
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from models import load_model
from models.encoders_2d import load_2d_model
from utils.cuda_utils.raycast_image import Project2DFeaturesCUDA
from utils.utils import load_state_with_same_shape
from utils.freemask_utils import *
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
from constants.dataset_sets import VAL_SCENES, TRAIN_SCENES

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import multiprocessing as mp

import sys
sys.path.append('./ext/CutLER/cutler')
sys.path.append('./ext/CutLER/cutler/demo')
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# define hard coded paths
config_path = 'CutLER/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml'
model_path = 'CutLER/models/cutler_cascade_final.pth'

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
                                         num_workers=1, augment_data=False,
                                         shuffle=False, repeat=False, batch_size=1)
    dataset = data_loader.dataset

    # Load all scenes
    eval_scenes = VAL_SCENES

    # Add instance segmentation
    evaluator = InstanceEvaluator(['foreground'], [1])

    # load config and model
    logging.info('===> Initializing CutLER model')
    pts = ['MODEL.WEIGHTS', model_path]
    cfg = setup_cfg(config_path,  ['MODEL.WEIGHTS', model_path])
    demo = VisualizationDemo(cfg)

    # Load model for single image projection
    demo = VisualizationDemo(cfg)

    # create directory to save outputs
    if not os.path.exists('outputs/eval_cutler'):
        os.makedirs('outputs/eval_cutler')

    # Start main iteration over the dataset
    tqdm_eval_scenes = tqdm(eval_scenes)
    for s_id, scene_name in enumerate(tqdm_eval_scenes):

        # Update tqdm description with scene name
        tqdm_eval_scenes.set_description(f'Processing {scene_name} ...')

        # load images
        RGB_PATH = config.data.scannet_images_path + f'/{scene_name}/color/'
        rgb_paths = sorted(glob.glob(RGB_PATH + '*.jpg'))
        rgb_image_ids = [int(os.path.basename(rgb_path).split('.')[0]) for rgb_path in rgb_paths]
        rgb_images = np.array([dataset.load_image(rgb_path, tuple(reversed(config.image_data.image_resolution))) for rgb_path in rgb_paths])

        # check if output path exists, create if not
        out_base_path = os.path.join(config.out_base_path, scene_name)
        if not os.path.exists(out_base_path):
            os.makedirs(out_base_path)

        # Check if all the files succesfully saved in out directory, if so skip
        if len(glob.glob(out_base_path + '/*.npy')) == len(rgb_image_ids):
            continue

        # do prediction and iteration for all images
        for img_id, s_image in enumerate(rgb_images):

            out_fname = os.path.join(out_base_path, f'{scene_name}_{rgb_image_ids[img_id]}_cutler.npy')

            # predict with image
            img_np = s_image.astype(np.uint8)
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

                scores += [score.item()]
                masks += [mask.cpu().numpy()]

            # Reverse both lists to get the lowest score first (override always with higher confidence)
            scores = scores[::-1]
            masks = masks[::-1]

            # Initialize combined mask and add all masks as instance ids
            combined_mask = np.zeros(img_shape, dtype=int) - 1
            for mask_id, m in enumerate(masks):
                combined_mask[m] = mask_id + 1

            # Save combined mask and visualized image
            np.save(out_fname, combined_mask)
            visualized_output.save(out_fname.replace('.npy', '.jpg'))


if __name__ == "__main__":
    main()

