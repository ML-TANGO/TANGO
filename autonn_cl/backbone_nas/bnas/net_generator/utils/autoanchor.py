# References:
# [1] YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

from .general import LOGGER

PREFIX = 'AutoAnchor: '


def check_anchor_order(m):
    '''
    Check anchor order against stride order for YOLOv5
    Detect() module m, and correct if necessary
    '''
    # mean anchor area per output layer
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(
            '%sReversing anchor order', PREFIX)
        m.anchors[:] = m.anchors.flip(0)
    return m
