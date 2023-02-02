'''
utils for loading dataset
'''

import contextlib
import glob
import hashlib
import os
import random
import sys
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, distributed, dataloader
from tqdm import tqdm

# net_generator Path 등록
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.augmentations import letterbox
from utils.general import (LOGGER, NUM_THREADS, xywhn2xyxy,
                            xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
# include image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format


def load_dataset(path,
                imgsz,
                batch_size,
                stride,
                cache=False,
                pad=0.0,
                rect=False,
                rank=-1,
                workers=8,
                image_weights=False,
                quad=False,
                prefix='',
                shuffle=False):
    '''
    load dataset
    '''
    if rect and shuffle:
        shuffle = False

    # init dataset *.cache only once if DDP
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            rect=rect,  # rectangular batches
            cache_images=cache,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size >
               1 else 0, workers])  # number of workers
    sampler = None if rank == - \
        1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    if quad:
        _fn = LoadImagesAndLabels.collate_fn4
    else:
        _fn = LoadImagesAndLabels.collate_fn
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_hash(paths):
    '''
    Returns a single hash value of a list of paths (files or dirs)
    '''
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s

class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler:
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class LoadImagesAndLabels(Dataset):
    '''
    YOLOv5 train_loader/val_loader,
    loads images and labels for training and validation
    '''
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                           cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 stride=32,
                 pad=0.0,
                 prefix=''):

        self.img_size = img_size
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.stride = stride
        self.path = path
        self.pad = pad
        self.batch_size = batch_size

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}')

        self.read_cache(p, prefix)

        # Update labels
        # filter labels to include only these classes (optional)
        include_class = []
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, label in enumerate(self.labels):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]

        # Rectangular Training
        if self.rect:
            self.set_rect()

        # Cache images into RAM/disk for faster training
        # (WARNING: large datasets may exceed system resources)
        self.ims = [None] * self.n
        self.npy_files = [Path(_f).with_suffix('.npy') for _f in self.im_files]
        if cache_images:
            _gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = \
                [None] * self.n, [None] * self.n
            fcn = (self.cache_images_to_disk
                   if cache_images == 'disk' else self.load_image)
            results = ThreadPool(NUM_THREADS).imap(fcn, range(self.n))
            pbar = tqdm(enumerate(results), total=self.n,
                        bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, _x in pbar:
                if cache_images == 'disk':
                    _gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], \
                        self.im_hw[i] = _x
                    _gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images \
                    ({_gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def read_cache(self, p, prefix):
        '''
        read cache
        '''
        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache = np.load(cache_path, allow_pickle=True).item()  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache = self.cache_labels(cache_path, prefix)  # run cache ops

        # Read cache
        _ = [cache.pop(k) for k in ('hash', 'version')]  # remove items
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / self.batch_size).astype(int)  # batch index
        self.nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

    def set_rect(self):
        '''
        set on Rectangular Training
        '''
        # Sort by aspect ratio
        _s = self.shapes  # wh
        _ar = _s[:, 1] / _s[:, 0]  # aspect ratio
        irect = _ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = _s[irect]  # wh
        _ar = _ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * self.nb
        for i in range(self.nb):
            ari = _ar[self.batch == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) *
                                           self.img_size /
                                           self.stride +
                                           self.pad).astype(np.int)
        self.batch_shapes *= self.stride

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.im_files, self.label_files), desc='Scanning images', total=len(self.im_files))
        for (img, label) in pbar:
            l = []
            image = Image.open(img)
            image.verify()  # PIL verify
            # _ = io.imread(img)  # skimage verify (from skimage import io)
            shape = exif_size(image)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
            if os.path.isfile(label):
                with open(label, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
            if len(l) == 0:
                l = np.zeros((0, 5), dtype=np.float32)
            x[img] = [l, shape]
        # nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        # desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        # with Pool(NUM_THREADS) as pool:
        #     pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
        #                 desc=desc,
        #                 total=len(self.im_files),
        #                 bar_format=BAR_FORMAT)
        #     for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
        #         nm += nm_f
        #         nf += nf_f
        #         ne += ne_f
        #         nc += nc_f
        #         if im_file:
        #             x[im_file] = [lb, shape, segments]
        #         if msg:
        #             msgs.append(msg)
        #         pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        # Load image
        img, (_h0, _w0), (_h, _w) = self.load_image(index)

        # Letterbox
        # final letterboxed shape
        shape = (self.batch_shapes[self.batch[index]]
                 if self.rect else self.img_size)
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=False)
        # for COCO mAP rescaling
        shapes = (_h0, _w0), ((_h / _h0, _w / _w0), pad)

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], ratio[0] * _w, ratio[1] * _h,
                padw=pad[0], padh=pad[1])

        _nl = len(labels)  # number of labels
        if _nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], _w=img.shape[1],
                _h=img.shape[0], clip=True, eps=1E-3)

        labels_out = torch.zeros((_nl, 6))
        if _nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        '''
        Loads 1 image from dataset index 'i',
        returns (im, original hw, resized hw)
        '''
        _im, _f, _fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if _im is None:  # not cached in RAM
            if _fn.exists():  # load npy
                _im = np.load(_fn)
            else:  # read image
                _im = cv2.imread(_f)  # BGR
                assert _im is not None, f'Image Not Found {_f}'
            _h0, _w0 = _im.shape[:2]  # orig hw
            _r = self.img_size / max(_h0, _w0)  # ratio
            if _r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (_r > 1) else cv2.INTER_AREA
                _im = cv2.resize(_im, (int(_w0 * _r), int(_h0 * _r)),
                                 interpolation=interp)
            # im, hw_original, hw_resized
            return _im, (_h0, _w0), _im.shape[:2]
        # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images_to_disk(self, i):
        '''
        Saves an image as an *.npy file for faster loading
        '''
        _f = self.npy_files[i]
        if not _f.exists():
            np.save(_f.as_posix(), cv2.imread(self.im_files[i]))

    @staticmethod
    def collate_fn(batch):
        '''
        collate_fn
        '''
        img, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        '''
        collate_fn4
        '''
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.Tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.Tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.Tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                img = F.interpolate(im[i].unsqueeze(0).float(),
                                    scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im[i].type())
                lb = label[i]
            else:
                img = torch.cat(
                    (torch.cat((im[i], im[i + 1]), 1),
                     torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat(
                    (label[i], label[i + 1] + ho,
                     label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(img)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
