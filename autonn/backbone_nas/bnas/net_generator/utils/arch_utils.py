'''
arch utils
'''

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None):
    '''
    set batch nomarization param
    '''
    replace_bn_with_gn(net, gn_channel_per_group)

    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum
            m.eps = eps
        elif isinstance(m, nn.GroupNorm):
            m.eps = eps

    return replace_conv2d_with_my_conv2d(net, ws_eps)


def get_bn_param(net):
    '''
    get bn param
    '''
    ws_eps = None
    for m in net.modules():
        if isinstance(m, MyConv2d):
            ws_eps = m.WS_EPS
            break
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return {
                "momentum": m.momentum,
                "eps": m.eps,
                "ws_eps": ws_eps,
            }
        if isinstance(m, nn.GroupNorm):
            return {
                "momentum": None,
                "eps": m.eps,
                "gn_channel_per_group": m.num_channels // m.num_groups,
                "ws_eps": ws_eps,
            }
    return None


def get_same_padding(kernel_size):
    '''
    get padding
    '''
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        _p1 = get_same_padding(kernel_size[0])
        _p2 = get_same_padding(kernel_size[1])
        return _p1, _p2
    assert isinstance(
        kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    '''
    replace conv
    '''
    if ws_eps is None:
        return

    for m in net.modules():
        to_update_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                # only replace conv2d layers
                # that are followed by normalization layers (i.e., no bias)
                to_update_dict[name] = sub_module
        for name, sub_module in to_update_dict.items():
            m.modules[name] = MyConv2d(
                sub_module.in_channels,
                sub_module.out_channels,
                sub_module.kernel_size,
                sub_module.stride,
                sub_module.padding,
                sub_module.dilation,
                sub_module.groups,
                sub_module.bias,
            )
            # load weight
            m.modules[name].load_state_dict(sub_module.state_dict())
            # load requires_grad
            m.modules[name].weight.requires_grad = \
                sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m.modules[name].bias.requires_grad = \
                    sub_module.bias.requires_grad
    # set ws_eps
    for m in net.modules():
        if isinstance(m, MyConv2d):
            m.WS_EPS = ws_eps


def min_divisible_value(_n1, _v1):
    """
    make sure v1 is divisible by n1, otherwise decrease v1
    """
    if _v1 >= _n1:
        return _n1
    while _n1 % _v1 != 0:
        _v1 -= 1
    return _v1


def replace_bn_with_gn(model, gn_channel_per_group):
    '''
    replace bn
    '''
    if gn_channel_per_group is None:
        return

    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                num_groups = sub_m.num_features // min_divisible_value(
                    sub_m.num_features, gn_channel_per_group
                )
                gn_m = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,
                )

                # load weight
                gn_m.weight.data.copy_(sub_m.weight.data)
                gn_m.bias.data.copy_(sub_m.bias.data)
                # load requires_grad
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = gn_m
        m.modules.update(to_replace_dict)


class MyConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(MyConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.ws_eps = None

    def weight_standardization(self, weight):
        '''
        weight_standardization
        '''
        if self.ws_eps is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.ws_eps
            )
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, _x):
        '''forward'''
        # if self.WS_EPS is None:
        #    return super(MyConv2d, self).forward(_x)
        return F.conv2d(
            _x,
            self.weight_standardization(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def __repr__(self):
        return super(MyConv2d, self).__repr__()[:-1] + \
            ", ws_eps=%s)" % self.WS_EPS
