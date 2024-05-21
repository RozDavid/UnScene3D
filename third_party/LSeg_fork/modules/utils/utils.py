import numpy as np
import logging

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from data.constants.scannet_constants import SCANNET_COLOR_MAP_LONG


def print_info(iteration,
               max_iteration,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               class_names=None):

    debug_str = "\n{}/{}: ".format(iteration, max_iteration)

    acc = (hist.diagonal() / (hist.sum(1) + 10e-5) * 100)
    debug_str += "\tAVG Loss {loss:.3f}\t" \
                 "AVG Score {top1:.3f}\t" \
                 "mIOU {mIOU:.3f} \t mAcc {mAcc:.3f}\n".format(
        loss=losses.item(), top1=scores.item(), mIOU=np.nanmean(ious), mAcc=np.nanmean(acc))

    if class_names is not None:
        debug_str += "\nClasses: " + ", ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ', '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAcc: ' + ', '.join('{:.03f}'.format(i) for i in acc) + '\n'

    logging.info(debug_str)

def nanmean_t(torch_array):
    value = torch_array[~torch.isnan(torch_array)].mean().item()
    if np.isnan(value):
        return 0.
    else:
        return value

class MetricAverageMeter(Metric):
    def __init__(self, ignore_index=-1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("value", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ignore_index = ignore_index

    def update(self, value, count: int = None):
        if count is not None:
            self.total += count
            self.value += torch.tensor(value * count).to(self.device)
        else:
            self.total += 1
            self.value += torch.tensor(value).to(self.device)

    def compute(self):
        return self.value / self.total


colorizer = np.vectorize(lambda i: list(SCANNET_COLOR_MAP_LONG[i]) if i in SCANNET_COLOR_MAP_LONG else np.array([0., 0., 0.]), otypes=[np.ndarray])
def colorize_image(image, dataset=None):

    color_img = np.zeros((*image.shape, 3))
    for i in range(color_img.shape[0]):
        for j in range(color_img.shape[1]):
            p = image[i, j]
            color_img[i, j, :] = list(SCANNET_COLOR_MAP_LONG[p + 1]) if p in SCANNET_COLOR_MAP_LONG else [0., 0., 0.]

    return color_img.astype(int)


