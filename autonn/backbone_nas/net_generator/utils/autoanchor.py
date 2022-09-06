# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

from .general import LOGGER

PREFIX = 'AutoAnchor: '


def check_anchor_order(_m):
    '''
    Check anchor order against stride order for YOLOv5
    Detect() module m, and correct if necessary
    '''
    # mean anchor area per output layer
    _a = _m.anchors.prod(-1).mean(-1).view(-1)
    _da = _a[-1] - _a[0]  # delta a
    _ds = _m.stride[-1] - _m.stride[0]  # delta s
    if _da and (_da.sign() != _ds.sign()):  # same order
        LOGGER.info(
            '%sReversing anchor order', PREFIX)
        _m.anchors[:] = _m.anchors.flip(0)
    return _m
