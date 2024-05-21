# +
# revised from https://github.com/zhanghang1989/PyTorch-Encoding/blob/331ecdd5306104614cb414b16fbcd9d1a8d40e1e/scripts/prepare_ade20k.py

"""Prepare ADE20K dataset"""
import os
import shutil
import argparse
import zipfile
from encoding.utils import download, mkdir
# -

_TARGET_DIR = '/mnt/data/Datasets/'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python prepare_ade20k.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
    download_dir = path
    mkdir(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    download_ade(_TARGET_DIR, overwrite=False)
