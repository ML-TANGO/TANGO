import os
import shutil

from pathlib import Path
from itertools import repeat
from zipfile import ZipFile
from multiprocessing.pool import ThreadPool

# download from pytorch hub (optional)
import torch

DATASETS_DIR = Path("/Data/coco128")


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1,
             retry=3):
    """ Multi-threaded file download and unzip function,
        used in dataset.yaml for autodownload
    """
    def download_one(url, dir):
        """ Download 1 file """
        success = True
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            for i in range(retry + 1):
                if curl:
                    s = 'sS' if threads > 1 else ''  # silent
                    r = os.system(f"curl -{s}L '{url}' -o"
                                  f" '{f}' --retry 9 -C -")  # curl download
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(
                        url, f,
                        progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    print(f'Download failure,'
                          f' retrying {i + 1}/{retry} {url}...')
                else:
                    print(f'Failed to download {url}...')

        if unzip and success and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    print(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        # multi-threaded
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


if __name__ == "__main__":
    # Download labels for YOLO-style Dectectors from YOLO V.5 github
    labels_url = \
        ['https://github.com/ultralytics/yolov5/releases/download/'
         + 'v1.0/coco2017labels.zip']
    download(labels_url, dir=DATASETS_DIR.parent)
    shutil.rmtree(str(DATASETS_DIR / 'annotations'))
    os.remove(str(DATASETS_DIR / 'LICENSE'))
    os.remove(str(DATASETS_DIR / 'README.txt'))

    # Download COCO images from official web sites
    # train2017 19G, 118k images
    # val2017 # 1G, 5k images
    # test2017 # 7G, 41k images (optional)
    images_urls = ['http://images.cocodataset.org/zips/train2017.zip',
                   'http://images.cocodataset.org/zips/val2017.zip',
                   'http://images.cocodataset.org/zips/test2017.zip']
    download(images_urls, dir=DATASETS_DIR / 'images', threads=3)
