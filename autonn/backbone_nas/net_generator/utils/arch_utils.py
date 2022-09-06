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

    for _m in net.modules():
        if isinstance(_m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            _m.momentum = momentum
            _m.eps = eps
        elif isinstance(_m, nn.GroupNorm):
            _m.eps = eps

    return replace_conv2d_with_my_conv2d(net, ws_eps)


def get_bn_param(net):
    '''
    get bn param
    '''
    ws_eps = None
    for _m in net.modules():
        if isinstance(_m, MyConv2d):
            ws_eps = _m.WS_EPS
            break
    for _m in net.modules():
        if isinstance(_m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return {
                "momentum": _m.momentum,
                "eps": _m.eps,
                "ws_eps": ws_eps,
            }
        if isinstance(_m, nn.GroupNorm):
            return {
                "momentum": None,
                "eps": _m.eps,
                "gn_channel_per_group": _m.num_channels // _m.num_groups,
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

    for _m in net.modules():
        to_update_dict = {}
        for name, sub_module in _m.named_children():
            if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                # only replace conv2d layers
                # that are followed by normalization layers (i.e., no bias)
                to_update_dict[name] = sub_module
        for name, sub_module in to_update_dict.items():
            _m.modules[name] = MyConv2d(
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
            _m.modules[name].load_state_dict(sub_module.state_dict())
            # load requires_grad
            _m.modules[name].weight.requires_grad = \
                sub_module.weight.requires_grad
            if sub_module.bias is not None:
                _m.modules[name].bias.requires_grad = \
                    sub_module.bias.requires_grad
    # set ws_eps
    for _m in net.modules():
        if isinstance(_m, MyConv2d):
            _m.WS_EPS = ws_eps


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

    for _m in model.modules():
        to_replace_dict = {}
        for name, sub_m in _m.named_children():
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
        _m.modules.update(to_replace_dict)


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


class MyModule(nn.Module):
    '''Mymodule'''

    def forward(self, _x):
        """forward"""
        raise NotImplementedError

    @property
    def module_str(self):
        '''module_str'''
        raise NotImplementedError

    @property
    def config(self):
        '''config'''
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        '''build config'''
        raise NotImplementedError


class MyNetwork(MyModule):
    '''MyNetwork'''
    CHANNEL_DIVISIBLE = 8

    def forward(self, x):
        """forward"""
        raise NotImplementedError

    @property
    def module_str(self):
        '''module_str'''
        raise NotImplementedError

    @property
    def config(self):
        '''config'''
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        '''build config'''
        raise NotImplementedError

    def zero_last_gamma(self):
        '''zero last gamma'''
        raise NotImplementedError

    @property
    def grouped_block_index(self):
        '''grouped b ind'''
        raise NotImplementedError

    def get_parameters(self, keys=None, mode="include"):
        '''get params'''
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        '''weight_param'''
        return self.get_parameters()


class MyGlobalAvgPool2d(nn.Module):
    '''MyGlobalAvgPool2d'''

    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, _x):
        '''forward'''
        return _x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "MyGlobalAvgPool2d(keep_dim=%s)" % self.keep_dim


def make_divisible(_v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(_v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * _v:
        new_v += divisor
    return new_v


class Hswish(nn.Module):
    '''Hswish activation func'''

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, _x):
        """forward"""
        return _x * F.relu6(_x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


class Hsigmoid(nn.Module):
    '''
    Hsigmoid
    '''

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, _x):
        """forward"""
        return F.relu6(_x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class SEModule(nn.Module):
    """SEModule"""
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(
            self.channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        self._fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(self.channel,
                                         num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv2d(num_mid,
                                         self.channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, _x):
        """forward"""
        _y = _x.mean(3, keepdim=True).mean(2, keepdim=True)
        _y = self._fc(_y)
        return _x * _y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


def build_activation(act_func, inplace=True):
    '''build_act'''
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    if act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    if act_func == "tanh":
        return nn.Tanh()
    if act_func == "sigmoid":
        return nn.Sigmoid()
    if act_func == "h_swish":
        return Hswish(inplace=inplace)
    if act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    if act_func is None or act_func == "none":
        return None
    raise ValueError("do not support: %s" % act_func)


class My2DLayer(MyModule):
    """My2DLayer"""

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=True,
            act_func="relu",
            dropout_rate=0,
            ops_order="weight_bn_act",
    ):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(
            self.act_func, self.ops_list[0] != "act" and self.use_bn
        )
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for _op in self.ops_list:
            if modules[_op] is None:
                continue
            if _op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(_op, modules[_op])

    @property
    def ops_list(self):
        '''ops list'''
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        """bn"""
        for _op in self.ops_list:
            if _op == "bn":
                return True
            if _op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_op(self):
        """weight op"""
        raise NotImplementedError

    def forward(self, x):
        """forward"""
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class ConvLayer(My2DLayer):
    """
    ConvLayer
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
            has_shuffle=False,
            use_se=False,
            use_bn=True,
            act_func="relu",
            dropout_rate=0,
            ops_order="weight_bn_act",
    ):
        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_se = use_se

        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn,
            act_func, dropout_rate, ops_order
        )
        if self.use_se:
            self.add_module("se", SEModule(self.out_channels))

    def weight_op(self):
        """weight_op"""
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    groups=min_divisible_value(self.in_channels, self.groups),
                    bias=self.bias,
                )
            }
        )

        return weight_dict

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedConv" % (
                    kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_Conv" % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = "%dx%d_DilatedGroupConv" % (
                    kernel_size[0], kernel_size[1])
            else:
                conv_str = "%dx%d_GroupConv" % (kernel_size[0], kernel_size[1])
        conv_str += "_O%d" % self.out_channels
        if self.use_se:
            conv_str = "SE_" + conv_str
        conv_str += "_" + self.act_func.upper()
        if self.use_bn:
            if isinstance(self.bn, nn.GroupNorm):
                conv_str += "_GN%d" % self.bn.num_groups
            elif isinstance(self.bn, nn.BatchNorm2d):
                conv_str += "_BN"
        return conv_str

    @property
    def config(self):
        return {
            "name": ConvLayer.__name__,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "has_shuffle": self.has_shuffle,
            "use_se": self.use_se,
            **super(ConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class IdentityLayer(My2DLayer):
    """IdentityLayer"""

    def __init__(
            self,
            in_channels,
            out_channels,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order="weight_bn_act",
    ):
        super(IdentityLayer, self).__init__(
            in_channels, out_channels, use_bn,
            act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        return None

    @property
    def module_str(self):
        return "Identity"

    @property
    def config(self):
        return {
            "name": IdentityLayer.__name__,
            **super(IdentityLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class ZeroLayer(MyModule):
    """ZeroLayer"""

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return "Zero"

    @property
    def config(self):
        return {
            "name": ZeroLayer.__name__,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer()


def set_layer_from_config(layer_config):
    """
    set_layer_from_config
    """
    if layer_config is None:
        return None
    return 1


class ResidualBlock(MyModule):
    """Res Block"""

    def __init__(self, conv, shortcut):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        '''forward'''
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        """module_str"""
        return "(%s, %s)" % (
            self.conv.module_str if self.conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None,
        )

    @property
    def config(self):
        """config"""
        return {
            "name": ResidualBlock.__name__,
            "conv": self.conv.config if self.conv is not None else None,
            "shortcut": (self.shortcut.config
                         if self.shortcut is not None else None),
        }

    @staticmethod
    def build_from_config(config):
        '''build config'''
        conv_config = (
            config["conv"]
            if "conv" in config
            else config["mobile_inverted_conv"]
        )
        conv = set_layer_from_config(conv_config)
        shortcut = set_layer_from_config(config["shortcut"])
        return ResidualBlock(conv, shortcut)

    @property
    def mobile_inverted_conv(self):
        """inverted"""
        return self.conv


class LinearLayer(MyModule):
    """LinearLayer"""

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            use_bn=False,
            act_func=None,
            dropout_rate=0,
            ops_order="weight_bn_act",
    ):
        super(LinearLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm1d(in_features)
            else:
                modules["bn"] = nn.BatchNorm1d(out_features)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(
            self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # linear
        modules["weight"] = {
            "linear": nn.Linear(self.in_features, self.out_features, self.bias)
        }

        # add modules
        for _op in self.ops_list:
            if modules[_op] is None:
                continue
            if _op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(_op, modules[_op])

    @property
    def ops_list(self):
        """ops_list"""
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        """bn_before_weight"""
        for _op in self.ops_list:
            if _op == "bn":
                return True
            if _op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def forward(self, x):
        """forward"""
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        return "%dx%d_Linear" % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            "name": LinearLayer.__name__,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)


class MBConvLayer(MyModule):
    '''
    MBConvLayer
    '''

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            expand_ratio=6,
            mid_channels=None,
            act_func="relu6",
            use_se=False,
            groups=None,
    ):
        super(MBConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.groups = groups

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                self.in_channels,
                                feature_dim, 1, 1, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        pad = get_same_padding(self.kernel_size)
        groups = (
            feature_dim
            if self.groups is None
            else min_divisible_value(feature_dim, self.groups)
        )
        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    pad,
                    groups=groups,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(self.act_func, inplace=True)),
        ]
        if self.use_se:
            depth_conv_modules.append(("se", SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim,
                                       out_channels, 1, 1, 0, bias=False)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = "%dx%d_MBConv%d_%s" % (
            self.kernel_size,
            self.kernel_size,
            expand_ratio,
            self.act_func.upper(),
        )
        if self.use_se:
            layer_str = "SE_" + layer_str
        layer_str += "_O%d" % self.out_channels
        if self.groups is not None:
            layer_str += "_G%d" % self.groups
        if isinstance(self.point_linear.bn, nn.GroupNorm):
            layer_str += "_GN%d" % self.point_linear.bn.num_groups
        elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
            layer_str += "_BN"

        return layer_str

    @property
    def config(self):
        return {
            "name": MBConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "groups": self.groups,
        }

    @staticmethod
    def build_from_config(config):
        return MBConvLayer(**config)
