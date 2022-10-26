# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from .general import (CONFIG_DIR, FONT, LOGGER, check_font,
                      clip_coords, increment_path, is_ascii, threaded,
                      xywh2xyxy, xyxy2xywh)
from .metrics import fitness

# Settings
RANK = int(os.environ.get('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    '''
    Ultralytics color palette https://ultralytics.com/
    '''

    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D',
                'CFD231', '48F90A', '92CC17', '3DDB86',
                '1A9334', '00D4BB', '2C99A8', '00C2FF',
                '344593', '6473FF', '0018EC', '8438FF',
                '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self._n = len(self.palette)

    def __call__(self, i, bgr=False):
        _c = self.palette[int(i) % self._n]
        return (_c[2], _c[1], _c[0]) if bgr else _c

    @staticmethod
    def hex2rgb(_h):
        '''rgb order (PIL)'''
        return tuple(int(_h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


COLORS = Colors()  # create instance for 'from utils.plots import colors'


def check_pil_font(font=FONT, size=10):
    '''
    Return a PIL TrueType Font,
    downloading to CONFIG_DIR if necessary
    '''
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(
            str(font) if font.exists() else font.name, size)
    except ValueError:  # download if missing
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        # except TypeError:
        #    check_requirements('Pillow>=8.4.0')
        # known issue https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # not online
            return ImageFont.load_default()


class Annotator:
    '''
    YOLOv5 Annotator for train/val mosaics and
    jpgs and detect/hub inference annotations
    '''

    def __init__(self, img, line_width=None,
                 font_size=None, font='Arial.ttf',
                 pil=False, example='abc'):
        assert img.data.contiguous, 'Image not contiguous. \
            Apply np.ascontiguousarray(im) to Annotator() input images.'
        # non-latin labels, i.e. asian, arabic, cyrillic
        non_ascii = not is_ascii(example)
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.img = img if isinstance(
                img, Image.Image) else Image.fromarray(img)
            self.draw = ImageDraw.Draw(self.img)
            self.font = check_pil_font(font='Arial.Unicode.ttf'
                                       if non_ascii
                                       else font,
                                       size=font_size or max(
                                           round(
                                               sum(self.img.size) / 2 * 0.035),
                                           12))
        else:  # use cv2
            self.img = img
        self._lw = line_width or max(
            round(sum(img.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='',
                  color=(128, 128, 128),
                  txt_color=(255, 255, 255)):
        '''
        Add one xyxy box to image with label
        '''
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self._lw, outline=color)  # box
            if label:
                _w, _h = self.font.getsize(label)  # text width, height
                outside = box[1] - _h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0],
                     box[1] - _h if outside else box[1], box[0] + _w + 1,
                     box[1] + 1 if outside else box[1] + _h + 1),
                    fill=color,
                )
                self.draw.text(
                    (box[0], box[1] - _h if outside else box[1]),
                    label, fill=txt_color, font=self.font)
        else:  # cv2
            _p1, _p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.img, _p1, _p2, color,
                          thickness=self._lw, lineType=cv2.LINE_AA)
            if label:
                _tf = max(self._lw - 1, 1)  # font thickness
                # text width, height
                _w, _h = cv2.getTextSize(
                    label, 0, fontScale=self._lw / 3, thickness=_tf)[0]
                outside = _p1[1] - _h >= 3
                _p2 = _p1[0] + _w, _p1[1] - _h - \
                    3 if outside else _p1[1] + _h + 3
                cv2.rectangle(self.img, _p1, _p2, color, -
                              1, cv2.LINE_AA)  # filled
                cv2.putText(self.img,
                            label, (_p1[0], _p1[1] -
                                    2 if outside else _p1[1] + _h + 2),
                            0,
                            self._lw / 3,
                            txt_color,
                            thickness=_tf,
                            lineType=cv2.LINE_AA)

    def rectangle(self, _xy, fill=None, outline=None, width=1):
        '''Add rectangle to image (PIL-only)'''
        self.draw.rectangle(_xy, fill, outline, width)

    def text(self, _xy, text,
             txt_color=(255, 255, 255)):
        '''Add text to image (PIL-only)'''
        _, _h = self.font.getsize(text)  # text width, height
        self.draw.text((_xy[0], _xy[1] - _h + 1), text,
                       fill=txt_color, font=self.font)

    def result(self):
        '''annotated image as array'''
        return np.asarray(self.img)


def feature_visualization(_x, module_type, stage,
                          _n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        _, channels, height, width = _x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            _f = save_dir / \
                f"stage{stage}_{module_type.split('.')[-1]}\
                    _features.png"  # filename

            # select batch index 0, block by channels
            blocks = torch.chunk(_x[0].cpu(), channels, dim=0)
            _n = min(_n, channels)  # number of plots
            _, _ax = plt.subplots(math.ceil(_n / 8), 8,
                                  tight_layout=True)  # 8 rows x n/8 cols
            _ax = _ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(_n):
                _ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                _ax[i].axis('off')

            LOGGER.info(
                'Saving %s... (%s/%s)',
                _f, _n, channels)
            plt.savefig(_f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(_f.with_suffix('.npy')),
                    _x[0].cpu().numpy())  # npy save


def hist2d(_x, _y, _n=100):
    '''
    2d histogram used in labels.png and evolve.png
    '''
    xedges, yedges = np.linspace(_x.min(),
                                 _x.max(),
                                 _n), np.linspace(_y.min(),
                                                  _y.max(), _n)
    hist, xedges, yedges = np.histogram2d(_x, _y, (xedges, yedges))
    xidx = np.clip(np.digitize(_x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(_y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data,
                            cutoff=1500,
                            _fs=50000,
                            order=5):
    '''
    https://stackoverflow.com/questions/\
        28536191/how-to-filter-smooth-with-scipy-numpy
    '''
    def butter_lowpass(cutoff, _fs, order):
        nyq = 0.5 * _fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    _b, _a = butter_lowpass(cutoff, _fs, order=order)
    return filtfilt(_b, _a, data)  # forward-backward filter


def output_to_target(output):
    '''
    Convert model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    '''
    targets = []
    for i, _o in enumerate(output):
        for *box, conf, cls in _o.cpu().numpy():
            targets.append(
                [i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


@threaded
def plot_images(images, targets, paths=None,
                fname='images.jpg', names=None,
                max_size=1920, max_subplots=16):
    '''
    Plot image grid with labels
    '''
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    _bs, _, _h, _w = images.shape  # batch size, _, height, width
    _bs = min(_bs, max_subplots)  # limit plot images
    _ns = np.ceil(_bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(_ns * _h), int(_ns * _w), 3),
                     255, dtype=np.uint8)  # init
    for i, _im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            _idx = i
            break
        _x, _y = int(_w * (i // _ns)), int(_h * (i % _ns))  # block origin
        _im = _im.transpose(1, 2, 0)
        mosaic[_y:_y + _h, _x:_x + _w, :] = _im

    # Resize (optional)
    scale = max_size / _ns / max(_h, _w)
    if scale < 1:
        _h = math.ceil(scale * _h)
        _w = math.ceil(scale * _w)
        mosaic = cv2.resize(mosaic, tuple(int(_x * _ns) for x in (_w, _h)))

    # Annotate
    _fs = int((_h + _w) * _ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(
        _fs / 10), font_size=_fs, pil=True, example=names)
    for i in range(_idx + 1):
        _x, _y = int(_w * (i // _ns)), int(_h * (i % _ns))  # block origin
        annotator.rectangle([_x, _y, _x + _w, _y + _h], None,
                            (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text(
                (_x + 5, _y + 5 + _h),
                text=Path(paths[i]).name[:40],
                txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            _ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(_ti[:, 2:6]).T
            classes = _ti[:, 1].astype('int')
            labels = _ti.shape[1] == 6  # labels if no conf column
            # check for confidence presence (label vs pred)
            conf = None if labels else _ti[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= _w  # scale to pixels
                    boxes[[1, 3]] *= _h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += _x
            boxes[[1, 3]] += _y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = COLORS(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.img.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler,
                      epochs=300, save_dir=''):
    '''
    Plot LR simulating training for full epochs
    '''
    optimizer, scheduler = copy(optimizer), copy(
        scheduler)  # do not modify originals
    _y = []
    for _ in range(epochs):
        scheduler.step()
        _y.append(optimizer.param_groups[0]['lr'])
    plt.plot(_y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_val_txt():
    '''
    Plot val.txt histograms
    '''
    _x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(_x[:, :4])
    _cx, _cy = box[:, 0], box[:, 1]

    _, _ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    _ax.hist2d(_cx, _cy, bins=600, cmax=10, cmin=0)
    _ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    _, _ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    _ax[0].hist(_cx, bins=600)
    _ax[1].hist(_cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)

# from utils.plots import *; plot_targets_txt()


def plot_targets_txt():
    '''
    Plot targets.txt histograms
    '''
    _x = np.loadtxt('targets.txt', dtype=np.float32).T
    _s = ['x targets', 'y targets', 'width targets', 'height targets']
    _, _ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    _ax = _ax.ravel()
    for i in range(4):
        _ax[i].hist(_x[i], bins=100,
                    label=f'{_x[i].mean():.3g} +/- {_x[i].std():.3g}')
        _ax[i].legend()
        _ax[i].set_title(_s[i])
    plt.savefig('targets.jpg', dpi=200)


# from utils.plots import *; plot_val_study()
def plot_val_study(file='', dirt='', _x=None):
    '''
    Plot file=study.txt generated by val.py
    (or plot all study*.txt in dir)
    '''
    save_dir = Path(file).parent if file else Path(dirt)
    plot2 = False  # plot additional results
    if plot2:
        _ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    _, _ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt'
    # for x in ['yolov5n6', 'yolov5s6',
    #               'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for _f in sorted(save_dir.glob('study*.txt')):
        _y = np.loadtxt(_f, dtype=np.float32, usecols=[
            0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        _x = np.arange(_y.shape[1]) if _x is None else np.array(_x)
        if plot2:
            _s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95',
                  't_preprocess (ms/img)',
                  't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                _ax[i].plot(_x, _y[i], '.-', linewidth=2, markersize=8)
                _ax[i].set_title(_s[i])

        _j = _y[3].argmax() + 1
        _ax2.plot(_y[5, 1:_j],
                  _y[3, 1:_j] * 1E2,
                  '.-',
                  linewidth=2,
                  markersize=8,
                  label=_f.stem.replace('study_coco_',
                                        '').replace('yolo', 'YOLO'))

    _ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]),
              [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
              'k.-',
              linewidth=2,
              markersize=8,
              alpha=.25,
              label='EfficientDet')

    _ax2.grid(alpha=0.2)
    _ax2.set_yticks(np.arange(20, 60, 5))
    _ax2.set_xlim(0, 57)
    _ax2.set_ylim(25, 55)
    _ax2.set_xlabel('GPU Speed (ms/img)')
    _ax2.set_ylabel('COCO AP val')
    _ax2.legend(loc='lower right')
    _f = save_dir / 'study.png'
    print(f'Saving {_f}...')
    plt.savefig(_f, dpi=300)


# @try_except
# @Timeout(30)
# def plot_labels(labels, names=(), save_dir=Path('')):
#     '''
#     known issue https://github.com/ultralytics/yolov5/issues/5395
#     known issue https://github.com/ultralytics/yolov5/issues/5611
#     '''
#     # plot dataset labels
#     LOGGER.info(
#         "Plotting labels to %s... ",
#         save_dir / 'labels.jpg')
#     _c, _b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
#     _nc = int(_c.max() + 1)  # number of classes
#     _x = pd.DataFrame(_b.transpose(), columns=['x', 'y', 'width', 'height'])

#     # seaborn correlogram
#     sn.pairplot(_x, corner=True, diag_kind='auto', kind='hist',
#                 diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
#     plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
#     plt.close()

#     # matplotlib labels
#     matplotlib.use('svg')  # faster
#     _ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
#     _y = _ax[0].hist(_c, bins=np.linspace(0, _nc, _nc + 1) - 0.5, rwidth=0.8)
#     try:  # color histogram bars by class
#         [_y[2].patches[i].set_color([_x / 255 for _x in colors(i)])
#          for i in range(_nc)]  # known issue #3195
#     except ValueError:
#         pass
#     _ax[0].set_ylabel('instances')
#     if 0 < len(names) < 30:
#         _ax[0].set_xticks(range(len(names)))
#         _ax[0].set_xticklabels(names, rotation=90, fontsize=10)
#     else:
#         _ax[0].set_xlabel('classes')
#     sn.histplot(_x, x='x', y='y', ax=_ax[2], bins=50, pmax=0.9)
#     sn.histplot(_x, x='width', y='height', ax=_ax[3], bins=50, pmax=0.9)

#     # rectangles
#     labels[:, 1:3] = 0.5  # center
#     labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
#     img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
#     for cls, *box in labels[:1000]:
#         ImageDraw.Draw(img).rectangle(
#             box, width=1, outline=colors(cls))  # plot
#     _ax[1].imshow(img)
#     _ax[1].axis('off')

#     for _a in [0, 1, 2, 3]:
#         for _s in ['top', 'right', 'left', 'bottom']:
#             _ax[_a].spines[_s].set_visible(False)

#     plt.savefig(save_dir / 'labels.jpg', dpi=200)
#     matplotlib.use('Agg')
#     plt.close()


# from utils.plots import *; plot_evolve()
def plot_evolve(evolve_csv='path/to/evolve.csv'):
    '''
    Plot evolve.csv hyp evolution results
    '''
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    _x = data.values
    _f = fitness(_x)
    _j = np.argmax(_f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {_j} of {evolve_csv}:')
    for i, _k in enumerate(keys[7:]):
        _v = _x[:, 7 + i]
        _mu = _v[_j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(_v, _f, c=hist2d(_v, _f, 20), cmap='viridis',
                    alpha=.8, edgecolors='none')
        plt.plot(_mu, _f.max(), 'k+', markersize=15)
        # limit to 40 characters
        plt.title(f'{_k} = {_mu:.3g}', fontdict={'size': 9})
        if i % 5 != 0:
            plt.yticks([])
        print(f'{_k:>15}: {_mu:.3g}')
    _f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(_f, dpi=200)
    plt.close()
    print(f'Saved {_f}')


def plot_results(file='path/to/results.csv', dirt=''):
    '''
    Plot training results.csv.
    Usage: from utils.plots import *;
    plot_results('path/to/results.csv')
    '''
    save_dir = Path(file).parent if file else Path(dirt)
    fig, _ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    _ax = _ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert files, f'No results.csv files found \
            in {save_dir.resolve()}, nothing to plot.'
    for _f in files:
        try:
            data = pd.read_csv(_f)
            _s = [x.strip() for x in data.columns]
            _x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                _y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                _ax[i].plot(_x, _y, marker='.', label=_f.stem,
                            linewidth=2, markersize=8)
                _ax[i].set_title(_s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except ValueError as _e:
            LOGGER.info(
                'Warning: Plotting error for %s: %s',
                _f, _e)
    _ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0,
                       labels=(), save_dir=''):
    '''
    Plot iDetection '*.txt' per-image logs.
    from utils.plots import *; profile_idetection()
    '''
    _ax = plt.subplots(2, 4, figsize=(12, 6),
                       tight_layout=True)[1].ravel()
    _s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery',
          'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for _fi, _f in enumerate(files):
        try:
            # clip first and last rows
            results = np.loadtxt(_f, ndmin=2).T[:, 90:-30]
            _n = results.shape[1]  # number of rows
            _x = np.arange(start, min(stop, _n) if stop else _n)
            results = results[:, _x]
            _t = (results[0] - results[0].min())  # set t0=0s
            results[0] = _x
            for i, _a in enumerate(_ax):
                if i < len(results):
                    label = (labels[_fi]
                             if labels
                             else _f.stem.replace('frames_', ''))
                    _a.plot(_t, results[i], marker='.',
                            label=label, linewidth=1, markersize=5)
                    _a.set_title(_s[i])
                    _a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        _a.spines[side].set_visible(False)
                else:
                    _a.remove()
        except ValueError as _e:
            print(f'Warning: Plotting error for {_f}; {_e}')
    _ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def save_one_box(xyxy, img, file=Path('im.jpg'),
                 gain=1.02, pad=10, square=False,
                 bgr=False, save=True):
    '''
    Save image crop as {file} with crop size multiple
    {gain} and {pad} pixels. Save and/or return crop
    '''
    xyxy = torch.Tensor(xyxy).view(-1, 4)
    _b = xyxy2xywh(xyxy)  # boxes
    if square:
        _b[:, 2:] = _b[:, 2:].max(1)[0].unsqueeze(
            1)  # attempt rectangle to square
    _b[:, 2:] = _b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(_b).long()
    clip_coords(xyxy, img.shape)
    crop = img[int(xyxy[0, 1]):int(xyxy[0, 3]),
               int(xyxy[0, 0]):int(xyxy[0, 2]),
               ::(1 if bgr else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        _f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        ).save(_f, quality=95, subsampling=0)
    return crop
