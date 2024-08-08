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
        # assert isinstance(self.runtime_depth, list)
        y, dt = [], []  # outputs
        elan_idx = 0
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                logger.info('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            if isinstance(m, ELAN) and isinstance(self.runtime_depth, list): # subnetwork run
                depth = self.runtime_depth[elan_idx]
                elan_idx += 1
                x = m(x, d=depth)
            else:
                x = m(x)  # fullnetwork run
            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def set_max_net(self):
        logger.info(f"... 1 ... set maximum depth for supernet")
        max_list = lambda x: [max(n) for n in x]
        self.set_active_subnet(d=max_list(self.depth_list))
        
    def set_active_subnet(self, d=None, **kwargs):
        logger.info(f"... 2 ... set ELANBlocks' depth {d}")
        self.runtime_depth = d   
        
    def sample_active_subnet(self):       
        # sample depth
        depth_setting = []
        for d_set in self.depth_list:
            d = random.choice(d_set)
            depth_setting.append(d)
        # set active subnet
        self.set_active_subnet(depth_setting) # ex) [3, 2, 3, 1, 4, 4, 1, 3]
        
        return {"d": depth_setting}
    
    def get_active_net_config(self): # self
        idx = 0
        d = deepcopy(self.yaml)
        
        # ch = 0
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            if 'ELAN' in m: # ELAN, ELANBlock, BBoneELAN, HeadELAN, ELAN2, TinyELAN
                args[-1] = self.runtime_depth[idx]
                idx += 1
            #     if 'BBone' in m:
            #         ch = int( args[0]*(args[-1]+1) )
            #     elif 'Head' in m:
            #         ch = int( (args[0]*2) + (args[0]/2*(args[-1]-1)) )
            # elif m == 'DyConv':
            #     if ch != 0:
            #         args = [ch, *args]
            #         ch = 0

        # [TENACE] inform depth of current subnet
        d['depth_list'] = self.runtime_depth
        # no more need depth_list(search space)
        # del d['depth_list']

        logger.info(f"......... done.") # f" subnet config = {d}")
        return d
    
    def get_active_subnet(self, preserve_weight=True):
        # create subnet
        config = self.get_active_net_config()

        logger.info(f"... 3 ... create this subnet model")
        #=======================================================================
        ch = config['ch']
        # subnet = YOLOModel(cfg=config, ch=[ch])
        subnet = Model(cfg=config, ch=[ch])
        #=======================================================================
        logger.info(f"......... done.")

        # subnet.info(verbose=True)

        logger.info(f"... 4 ... load pre-trained weights from supernet")
        dict_cfg = config['backbone'] + config['head']
        if preserve_weight:
            # extract ELANBlock & DyConv
            elan_idx = 0
            out_ch = 0
            model = deepcopy(self.model) # pre-trained supernet
            model.eval()
            for i, m in enumerate(model):
                if isinstance(m, ELAN): # ELAN, BBoneELAN, HeadELAN
                    logger.info(f"......... extract ELANblock {m.i} {m.f} {m.type} {m.np}")
                    depth = self.runtime_depth[elan_idx]
                    act_idx = m.act_idx[depth]
                    model[i] = ELANBlock(m.mode, deepcopy(m.layers[:act_idx+1]), depth)
                    np = sum([x.numel() for x in model[i].parameters()])  # re-calculate number params
                    model[i].i, model[i].f, model[i].type, model[i].np = m.i, m.f, m.type, np
                    logger.info(f"......... replace ELANblock {m.i} {m.f} {m.type} {np}")
                    # logger.info(f"......... and fill in weights from supernet")
                    elan_idx += 1
                    in_ch = dict_cfg[i][-1][0]
                    if m.mode == 'BBone':
                        elan_out_ch = int( in_ch * (depth+1) )
                    elif m.mode == 'Head':
                        elan_out_ch = int( in_ch*2 + (in_ch/2 * (depth-1)) )
                elif isinstance(m, DyConv):
                    logger.info(f"......... extract DyConv {m.i} {m.f} {m.type} {m.np}")
                    if elan_out_ch != 0:
                        model[i] = DyConvBlock(m.conv, m.bn, m.act, elan_out_ch)
                        elan_out_ch = 0
                    np = sum([x.numel() for x in model[i].parameters()])  # re-calculate number params
                    model[i].i, model[i].f, model[i].type, model[i].np = m.i, m.f, m.type, np
                    logger.info(f"......... replace DyConv {m.i} {m.f} {m.type} {np}")
                    # logger.info(f"......... and fill in weights from supernet")

            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

            # subnet.model.load_state_dict( model.state_dict() )
            subnet.model = model
            subnet.yaml = config
            del model

        logger.info(f"......... done.") #f" subnet model = {subnet.model}")
        return subnet
    

if __name__ == "__main__":
    profile=False
    device = select_device('0')
    
    # Create model
    supernet = NasModel(cfg='./yaml/yolov7_supernet.yml').to(device)
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
