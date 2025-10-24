"""
Download utils
"""

import logging
import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size
    # https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2,
    # checks and removes incomplete downloads < min_bytes
    from yolov5_utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist" \
                 f" or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file),
                                       progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, \
            assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # curl download, retry and resume on fail
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5'):
    """ Attempt file download if does not exist
        from utils.downloads import *; attempt_download() """
    from yolov5_utils.general import LOGGER

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        # decode '%2F' to '/' etc.
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            # parse authentication https://url.com/file.txt?auth...
            file = name.split('?')[0]
            if Path(file).is_file():
                # file already exists
                LOGGER.info(f'Found {url} locally at {file}')
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        # make parent dir (if required)
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # github api
            response = requests.get(f'https://api.github.com/repos/'
                                    f'{repo}/releases/latest').json()
            # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:  # fallback plan
            assets = [
                'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt',
                'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt',
                'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output(
                    'git tag', shell=True,
                    stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'v6.1'  # current release

        if name in assets:
            # backup gdrive mirror
            url3 = 'https://drive.google.com/drive/folders'
            url3 += '/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'
            safe_download(
                file,
                url=f'https://github.com/'
                    f'{repo}/releases/download/{tag}/{name}',
                url2=f'https://storage.googleapis.com/'
                     f'{repo}/{tag}/{name}',  # backup url (optional)
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading '
                          f'from https://github.com/{repo}/releases/'
                          f'{tag} or {url3}')

    return str(file)


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
