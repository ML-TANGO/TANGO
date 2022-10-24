'''
utils for loading dataset
'''

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
from easydict import EasyDict as edict
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, distributed
from tqdm import tqdm

# net_generator Path 등록
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.augmentations import letterbox
from utils.general import (LOGGER, NUM_THREADS, segments2boxes, xywhn2xyxy,
                            xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
# include image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'
# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format


def load_dataset(config):
    '''
    load dataset
    '''
    if config.rect and config.shuffle:
        shuffle = False

    # init dataset *.cache only once if DDP
    with torch_distributed_zero_first(config.rank):
        dataset = LoadImagesAndLabels(config)

    batch_size = min(config.batch_size, len(dataset))
    _nd = torch.cuda.device_count()  # number of CUDA devices
    _nw = min([os.cpu_count() // max(_nd, 1), batch_size if batch_size >
               1 else 0, config.workers])  # number of workers
    sampler = None if config.rank == - \
        1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader
    if config.quad:
        _fn = LoadImagesAndLabels.collate_fn4
    else:
        _fn = LoadImagesAndLabels.collate_fn
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=_nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=_fn), dataset


def img2label_paths(img_paths):
    '''
    Define label paths as a function of image paths
    /images/, /labels/ substrings
    '''
    _sa, _sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    paths = []
    for _x in img_paths:
        paths.append(_sb.join(_x.rsplit(_sa, 1)).rsplit('.', 1)[0])
    return paths


def get_hash(paths):
    '''
    Returns a single hash value of a list of paths (files or dirs)
    '''
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    _h = hashlib.md5(str(size).encode())  # hash sizes
    _h.update(''.join(paths).encode())  # hash paths
    return _h.hexdigest()  # return hash


def exif_size(img):
    '''
    Returns exif-corrected PIL size
    '''
    _s = img.size  # (width, height)
    for orient in ExifTags.TAGS:
        if ExifTags.TAGS[orient] == 'Orientation':
            orientation = orient
            break

    try:
        rotation = dict(img.getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            _s = (_s[1], _s[0])
    except IndexError as _e:
        pass

    return _s


class LoadImagesAndLabels(Dataset):
    '''
    YOLOv5 train_loader/val_loader,
    loads images and labels for training and validation
    '''
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                           cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self,
                 config):
        config.path = config.data_path
        self.config = config

        _p = self.check_img_files(config)

        self.read_cache(_p)

        # Update labels
        # filter labels to include only these classes (optional)
        include_class = []
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.config.labels,
                                                 self.config.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.config.labels[i] = label[j]
                if segment:
                    self.config.segments[i] = segment[j]

        # Rectangular Training
        if config.rect:
            self.set_rect()

        # Cache images into RAM/disk for faster training
        # (WARNING: large datasets may exceed system resources)
        self.ims = [None] * self.config.n
        self.npy_files = [Path(_f).with_suffix('.npy') for _f in self.im_files]
        if config.cache:
            _gb = 0  # Gigabytes of cached images
            self.config.im_hw0, self.config.im_hw = \
                [None] * self.config.n, [None] * self.config.n
            fcn = (self.cache_images_to_disk
                   if config.cache == 'disk' else self.load_image)
            results = ThreadPool(NUM_THREADS).imap(fcn, range(self.config.n))
            pbar = tqdm(enumerate(results), total=self.config.n,
                        bar_format=BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, _x in pbar:
                if config.cache == 'disk':
                    _gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    # im, hw_orig, hw_resized = load_image(self, i)
                    self.ims[i], self.config.im_hw0[i], \
                        self.config.im_hw[i] = _x
                    _gb += self.ims[i].nbytes
                pbar.desc = f'{config.prefix}Caching images \
                    ({_gb / 1E9:.1f}GB {config.cache})'
            pbar.close()

    def check_img_files(self, config):
        '''
        check image files
        '''
        try:
            _f = []  # image files
            _p = None
            for _p in (config.path
                       if isinstance(config.path, list) else [config.path]):
                _p = Path(_p)  # os-agnostic
                if _p.is_dir():  # dir
                    _f += glob.glob(str(_p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif _p.is_file():  # file
                    with open(_p) as _t:
                        _t = _t.read().strip().splitlines()
                        parent = str(_p.parent) + os.sep
                        # local to global path
                        _f += [_x.replace('./', parent)
                               if _x.startswith('./') else _x for _x in _t]
                        # local to global path (pathlib)
                        # f += [p.parent / x.lstrip(os.sep) for x in t]
                else:
                    raise Exception(f'{config.prefix}{_p} does not exist')
            self.im_files = sorted(_x.replace('/', os.sep)
                                   for _x in _f
                                   if _x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{config.prefix}No images found'
        except Exception as _e:
            raise Exception(
                f'{config.prefix}Error loading data \
                    from {config.path}: {_e}\nSee {HELP_URL}')
        return _p

    def read_cache(self, _p):
        '''
        read cache
        '''
        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (_p if _p.is_file() else Path(
            self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(
                cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(
                self.label_files + self.im_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path), False  # cache

        # Display cache
        # found, missing, empty, corrupt, total
        _nf, _nm, _ne, _nc, _n = cache.pop('results')
        if exists and LOCAL_RANK in {-1, 0}:
            _d = f"Scanning '{cache_path}' images and labels... \
                {_nf} found, {_nm} missing, {_ne} empty, {_nc} corrupt"
            tqdm(None, desc=self.config.prefix + _d, total=_n, initial=_n,
                 bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert _nf > 0, f'{self.config.prefix}No labels in {cache_path}. \
            Can not train without labels. See {HELP_URL}'

        # Read cache
        _ = [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.config.segments = zip(*cache.values())
        self.config.labels = list(labels)
        self.config.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        self.config.n = len(shapes)  # number of images
        self.config.batch = np.floor(np.arange(self.config.n) /
                                     self.config.batch_size).astype(np.int)
        self.indices = range(self.config.n)

    def set_rect(self):
        '''
        set on Rectangular Training
        '''
        # Sort by aspect ratio
        _s = self.config.shapes  # wh
        _ar = _s[:, 1] / _s[:, 0]  # aspect ratio
        irect = _ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.label_files = [self.label_files[i] for i in irect]
        self.config.labels = [self.config.labels[i] for i in irect]
        self.config.shapes = _s[irect]  # wh
        _ar = _ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * self.config.batch[-1] + 1
        for i in range(self.config.batch[-1] + 1):
            ari = _ar[self.config.batch == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.config.batch_shapes = np.ceil(np.array(shapes) *
                                           self.config.imgsz /
                                           self.config.stride +
                                           self.config.pad).astype(np.int)
        self.config.batch_shapes *= self.config.stride

    def cache_labels(self, path=Path('./labels.cache')):
        '''
        Cache dataset labels, check images and read shapes
        '''
        args = edict()
        args.cx = {}  # dict
        # number missing, found, empty, corrupt, messages
        args.nm, args.nf, args.ne, args.nc, args.msgs = 0, 0, 0, 0, []
        args.desc = f"{self.config.prefix}Scanning \
            '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label,
                                  zip(self.im_files,
                                      self.label_files,
                                      repeat(self.config.prefix))),
                        desc=args.desc,
                        total=len(self.im_files),
                        bar_format=BAR_FORMAT)
            for (im_file, _lb, shape, segments,
                 nm_f, nf_f, ne_f, nc_f, msg) in pbar:
                args.nm += nm_f
                args.nf += nf_f
                args.ne += ne_f
                args.nc += nc_f
                if im_file:
                    args.cx[im_file] = [_lb, shape, segments]
                if msg:
                    args.msgs.append(msg)
                pbar.desc = f"{args.desc}{args.nf} found, \
                    {args.nm} missing, {args.ne} empty, {args.nc} corrupt"

        pbar.close()
        if args.msgs:
            LOGGER.info('\n'.join(args.msgs))
        if args.nf == 0:
            LOGGER.warning('%sWARNING: No labels found in %s. See %s',
                           self.config.prefix, path, HELP_URL)
        args.cx['hash'] = get_hash(self.label_files + self.im_files)
        args.cx['results'] = (args.nf, args.nm, args.ne,
                              args.nc, len(self.im_files))
        args.cx['msgs'] = args.msgs  # warnings
        args.cx['version'] = self.cache_version  # cache version
        try:
            np.save(path, args.cx)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info("%sNew cache created: %s", self.config.prefix, path)
        except TypeError as _e:
            # not writeable
            LOGGER.warning("%sWARNING: Cache director %s is not writeable: %s",
                           self.config.prefix, path.parent, _e)
        return args.cx

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        # Load image
        img, (_h0, _w0), (_h, _w) = self.load_image(index)

        # Letterbox
        # final letterboxed shape
        shape = (self.config.batch_shapes[self.config.batch[index]]
                 if self.config.rect else self.config.imgsz)
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=False)
        # for COCO mAP rescaling
        shapes = (_h0, _w0), ((_h / _h0, _w / _w0), pad)

        labels = self.config.labels[index].copy()
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
            _r = self.config.imgsz / max(_h0, _w0)  # ratio
            if _r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (_r > 1) else cv2.INTER_AREA
                _im = cv2.resize(_im, (int(_w0 * _r), int(_h0 * _r)),
                                 interpolation=interp)
            # im, hw_original, hw_resized
            return _im, (_h0, _w0), _im.shape[:2]
        # im, hw_original, hw_resized
        return self.ims[i], self.config.im_hw0[i], self.config.im_hw[i]

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
        for i, _lb in enumerate(label):
            _lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        '''
        collate_fn4
        '''
        img, label, path, shapes = zip(*batch)  # transposed
        _n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:_n], shapes[:_n]

        _ho = torch.Tensor([[0.0, 0, 0, 1, 0, 0]])
        _wo = torch.Tensor([[0.0, 0, 1, 0, 0, 0]])
        _s = torch.Tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(_n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                img = F.interpolate(img[i].unsqueeze(0).float(),
                                    scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(img[i].type())
                _lb = label[i]
            else:
                img = torch.cat(
                    (torch.cat((img[i], img[i + 1]), 1),
                     torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                _lb = torch.cat(
                    (label[i], label[i + 1] + _ho,
                     label[i + 2] + _wo, label[i + 3] + _ho + _wo), 0) * _s
            im4.append(img)
            label4.append(_lb)

        for i, _lb in enumerate(label4):
            _lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


def verify_image_label(args):
    '''
    Verify one image-label pair
    '''
    ard = edict()
    ard.im_file, ard.lb_file, ard.prefix = args
    # number (missing, found, empty, corrupt), message, segments
    _nm, _nf, _ne, _nc, ard.msg, ard.segments = 0, 0, 0, 0, '', []

    try:
        # verify images
        img = Image.open(ard.im_file)
        img.verify()  # PIL verify
        shape = exif_size(img)  # image size
        assert (shape[0] > 9) & (
            shape[1] > 9), f'image size {shape} <10 pixels'
        assert img.format.lower(
        ) in IMG_FORMATS, f'invalid image format {img.format}'
        if img.format.lower() in ('jpg', 'jpeg'):
            with open(ard.im_file, 'rb') as _f:
                _f.seek(-2, 2)
                if _f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(ard.im_file)).save(
                        ard.im_file, 'JPEG', subsampling=0, quality=100)
                    ard.msg = f'{ard.prefix}WARNING: {ard.im_file}: \
                        corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(ard.lb_file):
            _nf = 1  # label found
            with open(ard.lb_file) as _f:
                _lb = [x.split()
                       for x in _f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in _lb):  # is segment
                    classes = np.array([x[0] for x in _lb], dtype=np.float32)
                    # (cls, xy1...)
                    ard.segments = [
                        np.array(_x[1:], dtype=np.float32).reshape(-1, 2)
                        for _x in _lb]
                    _lb = np.concatenate(
                        (classes.reshape(-1, 1),
                         segments2boxes(ard.segments)), 1)  # (cls, xywh)
                _lb = np.array(_lb, dtype=np.float32)
            _nl = len(_lb)
            if _nl:
                assert _lb.shape[1] == 5, f'labels require 5 columns,\
                    {_lb.shape[1]} columns detected'
                assert (_lb >= 0).all(), f'negative \
                    label values {_lb[_lb < 0]}'
                assert (_lb[:, 1:] <= 1).all(
                ), f'non-normalized or out of bounds \
                    coordinates {_lb[:, 1:][_lb[:, 1:] > 1]}'
                _, i = np.unique(_lb, axis=0, return_index=True)
                if len(i) < _nl:  # duplicate row check
                    _lb = _lb[i]  # remove duplicates
                    if ard.segments:
                        ard.segments = ard.segments[i]
                    ard.msg = f'{ard.prefix}WARNING: \
                        {ard.im_file}: {_nl - len(i)} \
                        duplicate labels removed'
            else:
                _ne = 1  # label empty
                _lb = np.zeros((0, 5), dtype=np.float32)
        else:
            _nm = 1  # label missing
            _lb = np.zeros((0, 5), dtype=np.float32)
        return (ard.im_file, _lb, shape, ard.segments,
                _nm, _nf, _ne, _nc, ard.msg)
    except Exception as _e:
        _nc = 1
        ard.msg = f'{ard.prefix}WARNING: {ard.im_file}: \
            ignoring corrupt image/label: {_e}'
        return [None, None, None, None, _nm, _nf, _ne, _nc, ard.msg]
