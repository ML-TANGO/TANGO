import logging
import random
from copy import deepcopy

import torch.nn as nn

from tango.common.models.yolo import Model
from tango.common.models.common import RepNCSPELAN4, RepNCSP

logger = logging.getLogger(__name__)


class NASModel(Model):
    """YOLOv9-based SuperNet with elastic RepNCSPELAN4 depths."""

    def __init__(self, cfg='yolov9-supernet.yml', ch=3, nc=None, anchors=None):
        self.runtime_depth = []
        super().__init__(cfg, ch, nc, anchors)

        backbone = self.yaml.get('backbone', [])
        head = self.yaml.get('head', [])
        self.depth_list = self.yaml.get('depth_list', [])
        self.depth_indices = [
            idx for idx, (f, n, m, args) in enumerate(backbone + head)
            if m == 'RepNCSPELAN4'
        ]

        if len(self.depth_list) != len(self.depth_indices):
            raise ValueError(
                f'depth_list length {len(self.depth_list)} must match RepNCSPELAN4 count {len(self.depth_indices)}'
            )

        self.set_max_net()

    def set_max_net(self):
        logger.info('NASModel: set maximum depth configuration for YOLOv9 supernet')
        self.set_active_subnet([max(d) for d in self.depth_list])

    def set_active_subnet(self, d=None, **kwargs):
        if d is None:
            raise ValueError('Active depth list must be provided')
        if len(d) != len(self.depth_list):
            raise ValueError(f'Expected {len(self.depth_list)} depth values, received {len(d)}')
        self.runtime_depth = d

    def sample_active_subnet(self):
        depth_setting = [random.choice(options) for options in self.depth_list]
        self.set_active_subnet(depth_setting)
        return {"d": depth_setting}

    def get_active_net_config(self):
        config = deepcopy(self.yaml)
        config['depth_list'] = list(self.runtime_depth)

        backbone_len = len(config['backbone'])
        combined = config['backbone'] + config['head']
        for depth_value, block_idx in zip(self.runtime_depth, self.depth_indices):
            block = combined[block_idx]
            args = block[3]
            if isinstance(args, list) and args:
                args[-1] = int(depth_value)

        config['backbone'] = combined[:backbone_len]
        config['head'] = combined[backbone_len:]
        return config

    def get_active_subnet(self, preserve_weight=True):
        config = self.get_active_net_config()
        ch = config.get('ch', self.yaml.get('ch', 3))
        subnet = Model(cfg=config, ch=[ch], nc=config.get('nc', self.nc), anchors=config.get('anchors', None))

        if preserve_weight:
            model = deepcopy(self.model)
            depth_iter = iter(self.runtime_depth)
            for module in model:
                if isinstance(module, RepNCSPELAN4):
                    depth = next(depth_iter)
                    self._adapt_repncsp(module.cv2[0], depth)
                    self._adapt_repncsp(module.cv3[0], depth)
            subnet.model = model
            subnet.yaml = config
        return subnet

    @staticmethod
    def _adapt_repncsp(block: RepNCSP, depth: int):
        children = list(block.m.children())
        max_depth = len(children)
        depth = max(1, min(int(depth), max_depth))
        block.m = nn.Sequential(*children[:depth])
        block.n = depth
