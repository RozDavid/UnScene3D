import sys
if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

import torch
from loguru import logger
import io
import json
import logging
import os
import errno
import pickle
import time
import string
import random

import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale
from torch import nn
from torchmetrics import Metric
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import collections
import wandb

def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_baseline_model(cfg, model):
    # if it is Minkoski weights
    cfg.model.in_channels = 3
    cfg.model.config.conv1_kernel_size = 5
    cfg.data.add_normals = False
    cfg.data.train_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.validation_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.test_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.voxel_size = 0.02
    model = model(cfg)
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    model.model.load_state_dict(state_dict)
    return cfg, model

def load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.backbone_checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(f"model.backbone.{key}", None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.backbone_checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if key.replace("model.backbone.", "") not in state_dict:
            logger.warning(
                f"{key} not in loaded checkpoint"
            )
            state_dict.update({key.replace("model.backbone.", ""): correct_dict[key]})
        elif state_dict[key.replace("model.backbone.", "")].shape != correct_dict[key].shape:
            logger.warning(
                f"incorrect shape {key}:{state_dict[key.replace('model.backbone.', '')].shape} vs {correct_dict[key].shape}"
            )
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if f"model.backbone.{key}" in correct_dict.keys():
            new_state_dict.update({f"model.backbone.{key}": state_dict[key]})
        elif key in correct_dict.keys():
            new_state_dict.update({key: correct_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model

def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parameters not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key, None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parameters have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if key not in state_dict:
            logger.warning(
                f"{key} not in loaded checkpoint"
            )
            state_dict.update({key: correct_dict[key]})
        elif state_dict[key].shape != correct_dict[key].shape:
            logger.warning(
                f"incorrect shape {key}:{state_dict[key].shape} vs {correct_dict[key].shape}"
            )
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if key in correct_dict.keys():
            new_state_dict.update({key: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model


def freeze_until(net, param_name: str = None):
    """
    Freeze net until param_name
    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD
    Args:
        net:
        param_name:
    Returns:
    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name

def load_state_with_same_shape(model, weights, prefix=''):

    model_state = model.state_dict()
    if list(weights.keys())[0].startswith('module.'):
        logging.info("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('model.'):
        logging.info("Loading Pytorch-Lightning weights from state")
        weights = {k.partition('model.')[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('encoder.'):
        logging.info("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition('encoder.')[2]: weights[k] for k in weights.keys()}

    if prefix != '':
        weights = {k.partition(prefix)[2]: weights[k] for k in weights.keys()}

    # This is when checkpoint containes the full plmodule
    target_keys = list(model_state.keys())
    if any(key.startswith('model_3d.') for key in target_keys) and any(key.startswith('model_2d.') for key in target_keys):
        logging.info("Loading full lightning module weights")
        bacbkbone_weights = np.array([key in target_keys for key in list(weights.keys())])
        bacbkbone_weights = np.array(list(weights.keys()))[bacbkbone_weights]
        weights = {k: weights[k] for k in bacbkbone_weights}

    # For continuous if keys containing model_3d - this is when loading 3d model weights only from full checkpoint
    if any(key.startswith('model_3d.') for key in list(weights.keys())) and not any(key.startswith('model_2d.') for key in target_keys):
        logging.info("Loading backbone weights starting with model_3d.")
        bacbkbone_weights = np.array([key.startswith('model_3d.') for key in list(weights.keys())])
        bacbkbone_weights = np.array(list(weights.keys()))[bacbkbone_weights]
        weights = {k.partition('model_3d.')[2]: weights[k] for k in bacbkbone_weights}

    # print(weights.items())
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print(model_state)

    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info(f"Loading weights for {len(filtered_weights.keys())}/{len(model_state.keys())} layers")

    return filtered_weights

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def load_image(path, scale=1.):
    image = Image.open(path)
    if scale != 1.:
        width, height = image.size
        target_shape = (int(width * scale), int(height * scale))
        image = image.resize(target_shape, Image.NEAREST)

    return np.array(image)

def precision_at_one(pred, target, ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return 0.


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def fast_hist_torch(pred, label, n):
    k = (label >= 0) & (label < n)
    return torch.bincount(n * label[k].int() + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def per_class_iu_torch(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))


def detect_blur_fft_batch(images, size=60):
    # convert to BW first
    if images.ndim > 3:
        gray = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images], axis=0 )
    else:
        gray = images

    # To frequency space
    h, w = gray.shape[1:]
    cx, cy = int(w / 2), int(h / 2)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    # inverse in window
    fft_shift[:, cy - size : cy + size, cx - size : cx + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    # return mean variance
    mag = 20 * np.log(np.abs(recon))
    mean = np.mean(mag, axis=(1, 2))
    return mean


def detect_blur_fft_batch_torch(images, size=30):
    # convert to BW first
    if images.ndim > 3:
        gray = rgb_to_grayscale(images).squeeze()
    else:
        gray = images

    # To frequency space
    h, w = gray.shape[1:]
    cx, cy = int(w / 2), int(h / 2)
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)

    # inverse in window
    fft_shift[:, cy - size: cy + size, cx - size: cx + size] = 0
    fft_shift = torch.fft.ifftshift(fft_shift)
    recon = torch.fft.ifft2(fft_shift)

    recon = 20 * torch.mean(torch.abs(recon), dim=(1, 2))

    return recon


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4/1) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            if im.shape[-1] == 4:
                # only render Alpha channel
                ax.imshow(im[..., 3])
            else: # depth only
                ax.imshow(im)
        if not show_axes:
            ax.set_axis_off()

    plt.show()

class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True, with_call=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        if with_call or self.calls == 0:
            self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
                            (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 10e-5)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def debug_on():
    import sys
    import pdb
    import functools
    import traceback

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def get_prediction(dataset, output, target):
    return output.max(1)[1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
    return torch.device('cuda' if is_cuda else 'cpu')

def randStr(chars = string.ascii_lowercase + string.digits, N=10):
    random.seed(time.time())
    return ''.join(random.choice(chars) for _ in range(N))

class HashTimeBatch(object):

    def __init__(self, prime=5279):
        self.prime = prime

    def __call__(self, time, batch):
        return self.hash(time, batch)

    def hash(self, time, batch):
        return self.prime * batch + time

    def dehash(self, key):
        time = key % self.prime
        batch = key / self.prime
        return time, batch


def save_obj(output_path, obj):
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(intput_path):
    with open(intput_path, 'rb') as f:
        return pickle.load(f)

def save_feature_maps(feature_maps, config, scene_name, targets=None, coords=None):

    base_file_name = '_'.join([scene_name, "feature_maps.pkl"])
    # Create directory to save visualization results.
    os.makedirs(config.test.visualize_path, exist_ok=True)

    out_dict = {'feature_map': feature_maps}
    if targets is not None:
        out_dict = {'target': targets, **out_dict}

    if coords is not None:
        out_dict = {'coords': coords, **out_dict}

    base_file_name = os.path.join(config.test.visualize_path, base_file_name)
    with open(base_file_name, 'wb') as f:
        pickle.dump(out_dict, f, pickle.HIGHEST_PROTOCOL)

def save_mean_features(mean_features, iteration, config):

    base_file_name = '_'.join([config.data.dataset, config.net.model, 'train_{}'.format(iteration), "mean_features.pkl"])

    # Create directory to save visualization results.
    os.makedirs(config.test.visualize_path, exist_ok=True)

    full_path = os.path.join(config.test.visualize_path, base_file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(mean_features, f, pickle.HIGHEST_PROTOCOL)

def nanmean_t(torch_array):
    value = torch_array[~torch.isnan(torch_array)].mean().item()
    if np.isnan(value):
        return 0.
    else:
        return value


def print_info(iteration,
               max_iteration,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               class_names=None,
               head_ious=None,
               common_ious=None,
               tail_ious=None,
               top_K_ious=None,
               top_K_accs=None,
               pixel_ious=None,
               pixel_accuracies=None,
               config=None):

    debug_str = "{}/{}: ".format(iteration, max_iteration)

    acc = (hist.diagonal() / hist.sum(1) * 100)
    debug_str += "\nAVG Loss {loss:.3f}\t" \
                 "AVG Score {top1:.2f}\t" \
                 "mIOU {mIOU:.2f} mAcc {mAcc:.2f}\n".format(loss=losses.item(), top1=scores.item(), mIOU=np.nanmean(ious), mAcc=np.nanmean(acc))

    if head_ious is not None:
         debug_str += 'Head mIoU {head:.2f} \t Common mIoU {common:.2f} \tTail mIoU {tail:.2f} \n'.format(head=head_ious, common=common_ious, tail=tail_ious)

    if class_names is not None:
        debug_str += "\nClasses: " + ", ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ', '.join('{:.02f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAcc: ' + ', '.join('{:.02f}'.format(i) for i in acc) + '\n'

    if (top_K_ious is not None) and top_K_accs is not None:
        debug_str += f'TOP {config.test.topk_metrics} Acc: ' + ', '.join('{:.02f}'.format(i) for i in top_K_accs) + '\n'
        debug_str += f'TOP {config.test.topk_metrics} IOU: ' + ', '.join('{:.02f}'.format(i) for i in top_K_ious) + '\n'

    if pixel_ious is not None:
        debug_str += 'Pixel IOU: ' + ', '.join('{:.02f}'.format(i) for i in pixel_ious) + '\n'
        debug_str += 'Pixel mAcc: ' + ', '.join('{:.02f}'.format(i) for i in pixel_accuracies) + '\n'

    logging.info(debug_str)

def plot_confusion_matrix(confusion_mat, class_names, title='Confusion Matrix'):
    data = []
    n_classes = confusion_mat.shape[0]

    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], confusion_mat[i, j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }
    return wandb.plot_table("wandb/confusion_matrix/v1",
                            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
                            fields,
                            {"title": title},)
