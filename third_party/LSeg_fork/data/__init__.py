import copy

import itertools
import functools
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import encoding.datasets as enc_ds
import data.scannet_dataset as scannet_dataset


encoding_datasets = {
    x: functools.partial(enc_ds.get_dataset, x)
    for x in ["coco", "ade20k", "pascal_voc", "pascal_aug", "pcontext", "citys"]
}

LOCAL_DATASETS = []
def add_datasets(module):
  LOCAL_DATASETS.extend([getattr(module, a) for a in dir(module) if 'Dataset' in a])

add_datasets(scannet_dataset)
m_dataset_dict = {dataset.__name__: dataset for dataset in LOCAL_DATASETS}

def get_dataset(name, **kwargs):
    if name in encoding_datasets:
        return encoding_datasets[name.lower()](**kwargs)
    elif name in m_dataset_dict:
        return load_dataset(name)(**kwargs)

    print(get_available_datasets())
    assert False, f"dataset {name} not found"


def get_available_datasets():
    return list(encoding_datasets.keys())  + [dset.__name__ for dset in LOCAL_DATASETS]

def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  if name not in m_dataset_dict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in LOCAL_DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = m_dataset_dict[name]
  return DatasetClass