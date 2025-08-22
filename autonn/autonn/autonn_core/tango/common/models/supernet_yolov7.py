## libraries in  yolov7 
import argparse
import logging
from copy import deepcopy

import torch

from tango.common.models.common import *
from tango.common.models.search_block import *
from tango.common.models.experimental import *
from tango.common.models.yolo import *
from tango.utils.autoanchor import check_anchor_order
from tango.utils.general import make_divisible, check_file, set_logging
from tango.utils.torch_utils import (   time_synchronized,
                                        fuse_conv_and_bn,
                                        model_info,
                                        scale_img,
                                        initialize_weights,
                                        select_device,
                                        copy_attr
                                    )
from tango.utils.loss import SigmoidBin

logger = logging.getLogger(__name__)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

## libraries from OFA code
import random

## libraries for type hint
from typing import List, Tuple, Union, Optional, Callable, Any

## Super class for YOLOSuperNet
# from .yolo_nas import *

## search_block.py
# from .search_block import ELAN, ELANBlock

_DEPTH_BLOCK_PREFIXES = (
    "RepNCSPELAN",   # RepNCSPELAN, RepNCSPELAN4 ...
    "BBoneELAN",
    "HeadELAN",
)

def _is_depth_block_yaml(m):
    name = m if isinstance(m, str) else getattr(m, "__name__", str(m))
    return any(name.startswith(p) for p in _DEPTH_BLOCK_PREFIXES)

def _is_depth_block_module(mod):
    name = mod.__class__.__name__
    return any(name.startswith(p) for p in _DEPTH_BLOCK_PREFIXES)

def _depth_arg_index(module_name: str, args: list) -> int:
    if module_name.startswith("BBoneELAN") or module_name.startswith("HeadELAN"):
        return 2
    if module_name.startswith("RepNCSPELAN"):
        return -1
    return None

class NASModel(Model):
    """ Create YOLOv7-based SuperNet 

    Args:
    -----------
    cfg: str
        path to yolo supernet yaml file
    ch: int
        number of input channels
    nc: int
        number of classes
    anchors: 2d-list
        anchors : np * 3(anchros/floor) * 2(width & height)

    Attributes:
    -----------
    model: nn.Sequential
        a sequence of nn.modules, i.e. YOLOSuperNet modules 
    save: list
        indice of jumping points to use for forward pass
    depth_list: list of int
        list of depth for each ELANBlock
    runtime_depth: list of int
        list of depth for each ELANBlock of subnetwork, but initialized int at first
    """     
    def __init__(
        self,
        cfg='yolov7-supernet.yml',
        ch=3, 
        nc=None, 
        anchors=None,
    ):
        self.runtime_depth = 0
        
        super(NASModel, self).__init__(cfg, ch, nc, anchors)
        
        self.depth_list = self.yaml['depth_list']
        self.set_max_net()
        
    def forward_once(self, x, profile=False):
        y, dt = [], []
        depth_idx = 0

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced and isinstance(m, _DEPTH_BLOCK_PREFIXES):
                break

            if profile:
                c = isinstance(m, _DEPTH_BLOCK_PREFIXES)
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                logger.info('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            if isinstance(self.runtime_depth, list) and _is_depth_block_module(m):
                depth = self.runtime_depth[depth_idx]
                depth_idx += 1
                if hasattr(m, 'forward_depth'):
                    x = m.forward_depth(x, depth)
                else:
                    try:
                        x = m(x, d=depth)
                    except TypeError:
                        if hasattr(m, 'set_active_depth'):
                            m.set_active_depth(depth)
                            x = m(x)
                        else:
                            x = m(x)
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def set_max_net(self):
        logger.info(f"... 1 ... set maximum depth for supernet")
        max_list = lambda x: [max(n) for n in x]
        self.set_active_subnet(d=max_list(self.depth_list))
        
    def set_active_subnet(self, d=None, **kwargs):
        yaml_blocks = self.yaml.get('backbone', []) + self.yaml.get('head', [])
        expected = sum(1 for (_, _, m, args) in yaml_blocks if _is_depth_block_yaml(m))

        if expected == 0:
            import warnings
            warnings.warn("[NAS] No depth-configurable blocks found in YAML; skipping depth injection.")
            self.runtime_depth = [] if not isinstance(d, list) else d[:0]
            return

        if d is None:
            if isinstance(self.depth_list, list) and all(isinstance(x, list) and x for x in self.depth_list):
                d = [max(x) for x in self.depth_list]
            else:
                raise ValueError("[NAS] Depth vector is None and depth_list is not a list-of-lists")

        if len(d) != expected:
            import warnings
            warnings.warn(f"[NAS] Depth vector length mismatch: got {len(d)} but model needs {expected}. "
                        f"Will auto-fix by padding/trimming.")
            if len(d) < expected:
                pad = []
                if isinstance(self.depth_list, list) and all(isinstance(x, list) and x for x in self.depth_list):
                    for i in range(len(d), expected):
                        pad.append(max(self.depth_list[i]))
                else:
                    pad = [d[-1] if d else 1] * (expected - len(d))
                d = d + pad
            else:
                d = d[:expected]

        self.runtime_depth = [int(x) for x in d]
        logger.info(f"... 2 ... set ELANBlocks' depth {self.runtime_depth}")
        
    def sample_active_subnet(self):       
        # sample depth
        depth_setting = []
        for d_set in self.depth_list:
            d = random.choice(d_set)
            depth_setting.append(d)
        # set active subnet
        self.set_active_subnet(depth_setting) # ex) [3, 2, 3, 1, 4, 4, 1, 3]
        
        return {"d": depth_setting}
    
    def get_active_net_config(self):
        idx = 0
        d = deepcopy(self.yaml)

        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            if _is_depth_block_yaml(m):
                name = m if isinstance(m, str) else getattr(m, "__name__", str(m))
                di = _depth_arg_index(name, args)
                if di is None or abs(di) > len(args):
                    logger.warning(f"[NAS] Cannot locate depth arg for {name}; skip depth injection.")
                else:
                    args[di] = int(self.runtime_depth[idx])
                    idx += 1

        d['depth_list'] = self.runtime_depth
        logger.info("......... done.")
        return d
    
    def get_active_subnet(self, preserve_weight=True):
        config = self.get_active_net_config()
        logger.info("... 3 ... create this subnet model")
        ch_val = config.get('ch', self.yaml.get('ch', 3))
        subnet = Model(
            cfg=config,
            ch=ch_val,
            nc=config.get('nc', None),
            anchors=config.get('anchors', None),
        )

        logger.info("......... done.")

        if preserve_weight:
            logger.info("... 4 ... load pre-trained weights from supernet")
            model = deepcopy(self.model).eval()
            depth_idx = 0

            for i, m in enumerate(model):
                if _is_depth_block_yaml(m):  # RepNCSPELAN*
                    depth = self.runtime_depth[depth_idx]
                    depth_idx += 1
                    if hasattr(m, 'truncate_to_depth'):
                        model[i] = m.truncate_to_depth(depth)
                    elif hasattr(m, 'export_subblock'):
                        model[i] = m.export_subblock(depth)
                    else:
                        if hasattr(m, 'set_active_depth'):
                            m.set_active_depth(depth)

            subnet.model = model
            subnet.yaml = config
            del model

        logger.info("......... done.")
        return subnet
    
if __name__ == "__main__":
    profile=False
    device = select_device('0')
    
    # Create model
    supernet = NASModel(cfg='./yaml/yolov7_supernet.yml').to(device)
    supernet.train()
    sample_depth_setting = supernet.sample_active_subnet()   
    subnet = supernet.get_active_subnet()
    print(sample_depth_setting)
    
    if profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = supernet(img, profile=profile)
        y = subnet(img, profile=profile)
    
    img = torch.rand(1, 3, 640, 640).to(device)
    y = supernet(img, profile=profile)
    y = subnet(img)
        
    sample_config = supernet.get_active_net_config()
    
    
    from yolo import parse_model
    ch = sample_config['ch']
    model, save = parse_model(sample_config, [ch])