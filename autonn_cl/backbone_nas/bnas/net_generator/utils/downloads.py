'''
data download functions
'''

import logging
import os
import subprocess
import sys
import urllib
from pathlib import Path

import requests
import torch

from .general import LOGGER

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir="~/.torch/", overwrite=False):
    '''
    download from once for all
    '''
    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write(
                'Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except ValueError as _e:
        # remove lock file so download can be executed next time.
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" %
                         url + "\n" + str(_e) + "\n")
        return None


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    '''
    Attempts to download file from url or url2,
    checks and removes incomplete downloads < min_bytes
    '''
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not \
        exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(
            "Downloading %s to %s...",
            url, file)
        torch.hub.download_url_to_file(
            url, str(file), progress=LOGGER.level <= logging.INFO)
        assert ((file.exists()
                 and file.stat().st_size >
                 min_bytes)), assert_msg  # check
    except ValueError as _e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        LOGGER.info(
            "ERROR: %s\nRe-attempting %s to %s...",
            _e, url2 or url, file)
        # curl download, retry and resume on fail
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            LOGGER.info(
                "ERROR: %s\n%s",
                assert_msg, error_msg)
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v6.1'):
    '''
    attempt download
    '''
    # Attempt file download from GitHub release assets
    # if not found locally. release = 'latest', 'v6.1', etc.

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v6.1') and
        # assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.1
        response = requests.get(
            f'https://api.github.com/repos/{repository}/\
                releases/{version}').json()  # github api
        # tag, assets
        return response['tag_name'], [x['name'] for x in response['assets']]
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
                LOGGER.info(
                    "Found %s locally at %s",
                    url, file)
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [
            'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt',
            'yolov5l.pt', 'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt',
            'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
        try:
            tag, assets = github_assets(repo, release)
        except ValueError:
            try:
                tag, assets = github_assets(repo)  # latest release
            except ValueError:
                try:
                    tag = subprocess.check_output(
                        'git tag',
                        shell=True,
                        stderr=subprocess.STDOUT).decode().split()[-1]
                except ValueError:
                    tag = release

        # make parent dir (if required)
        file.parent.mkdir(parents=True, exist_ok=True)
        if name in assets:
            # backup gdrive mirror
            url3 = 'https://drive.google.com/drive/\
                folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'
            safe_download(
                file,
                url=f'https://github.com/{repo}/\
                    releases/download/{tag}/{name}',
                # backup url (optional)
                url2=f'https://storage.googleapis.com/{repo}/{tag}/{name}',
                min_bytes=1E5,
                error_msg=f'{file} missing, try downloading from \
                    https://github.com/{repo}/releases/{tag} or {url3}')

    return str(file)
