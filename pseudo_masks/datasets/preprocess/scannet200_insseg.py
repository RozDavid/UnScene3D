import os
import plyfile
import json
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import itertools

# Add the root directory to the python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from datasets.scannet200.scannet200_constants import VALID_CLASS_IDS_200

# Load labels table
labels_pd = pd.read_csv('pseudo_masks/constants/scannetv2-labels.combined.tsv', sep='\t', header=0)
category_label_names = labels_pd['category'].unique()
valid_raw_cats = np.unique(labels_pd['raw_category'].to_numpy())

def RAW2SCANNET(label):
    if label not in valid_raw_cats:
        return 0

    label_id = int(labels_pd[labels_pd['raw_category'] == label]['id'].iloc[0])
    if not label_id in VALID_CLASS_IDS_200:
        label_id = 0
    return label_id

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./data/ScanNet/scans/')
    parser.add_argument('--output', default='./data/ScanNet/scannet_200_insseg')
    parser.add_argument('--num_threads', type=int, default=8)
    opt = parser.parse_args()
    return opt


def main(config, scene_name):

    print(scene_name)
    # Over-segmented segments: maps from segment to vertex/point IDs
    segid_to_pointid = {}
    segfile = os.path.join(config.input, scene_name, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))

    if not os.path.exists(segfile):  # test scene
        return

    with open(segfile) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    # Raw points in XYZRGBA
    ply_filename = os.path.join(config.input, scene_name, '%s_vh_clean_2.ply' % (scene_name))
    f = plyfile.PlyData().read(ply_filename)
    points = np.array([list(x) for x in f.elements[0]])

    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(config.input, scene_name, '%s.aggregation.json' % (scene_name))
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])

    # Each instance's points
    instance_labels = np.zeros(points.shape[0])
    semantic_labels = np.zeros(points.shape[0])
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        pointids = np.array(pointids)
        instance_labels[pointids] = i + 1
        semantic_labels[pointids] = RAW2SCANNET(labels[i])

    colors = points[:, 3:6]
    points = points[:, 0:3]  # XYZ+RGB+NORMAL
    torch.save((points, colors, semantic_labels, instance_labels), os.path.join(config.output, scene_name + '.pth'))


if __name__ == '__main__':
    config = parse_args()
    os.makedirs(config.output, exist_ok=True)

    # Before starting the parallel processing, we need copy the scene splits to the target directory
    shutil.copy('pseudo_masks/datasets/preprocess/splits/train.txt', config.output)
    shutil.copy('pseudo_masks/datasets/preprocess/splits/val.txt', config.output)
    shutil.copy('pseudo_masks/datasets/preprocess/splits/test.txt', config.output)

    # Parallel extract semantic instances in ScanNet scenes
    pool = ProcessPoolExecutor(max_workers=config.num_threads)
    result = list(pool.map(main, itertools.repeat(config), os.listdir(config.input)))
