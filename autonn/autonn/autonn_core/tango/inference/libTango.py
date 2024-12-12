import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import logging
logger = logging.getLogger(__name__)

# ======================== basic modules =======================================
# 1. Conv: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 2. Concat: torch.cat
# 3. ReOrg: torch.cat
# 4. MP: nn.MaxPool2d
# 5. SP: nn.MaxPool2d
# 6. Silence: -
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    

class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    
    def forward(self, x):
        return x
    

# ======================== resnet modules ======================================
# 1. BasicBlock: nn.Conv2d, nn.BatchNorm2d, nn.ReLU
# 2. Bottleneck: nn.Conv2d, nn.BatchNorm2d, nn.ReLU
# 3. cBasicBlock: nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, F.pad
from typing import Callable, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3c(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class cBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super(cBasicBlock, self).__init__()

        self.conv1 = conv3x3c(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3c(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = IdentityPadding(inplanes, planes, stride)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


# ======================== spp modules =========================================
# 1. DownC: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, nn.MaxPool2d, torch.cat
# 2. SPPCSPC: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, nn.MaxPool2d, torch.cat
# 3. SPPELAN: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, nn.MaxPool2d, torch.cat
class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)
    

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4*c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


# ======================== etc. modules ========================================
# 1. AConv: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, F.avg_pool2d
# 2. ADown: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, F.avg_pool2d, F.max_pool2d, torch.cat, torch.chunk
# 3. CBLiner: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, torch.split
# 4. CBFuse: F.interpolate, torch.sum, torch.stack
class AConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)
            
    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 1)
    
    def forward(self,x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


# ======================== elan modules ========================================
# 1. ELAN1: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, torch.chunk, torch.cat, torch.split
# 2. BBoneELAN: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, torch.cat
# 3. HeadELAN: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, torch.cat
# 4. TinyELAN: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, torch.cat
class ELAN1(nn.Module):
    def __init__(self, c1, c2, c3, c4):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3//2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN(nn.Module):
    def __init__(self, c1, c2, k, depth):
        super(ELAN, self).__init__()
        assert c1 % 2 == 0
        self.c2 = c2
        self.depth = depth


class BBoneELAN(ELAN):
    mode = 'BBone'
    def __init__(self, c1, c2, k, depth):
        super(BBoneELAN, self).__init__(c1, c2, k, depth)
        assert c1 % 2 == 0
        
        layers = []
        # make layers according to depth
        for i in range(depth):
            if i == 0: # depth 1
                layers.append(Conv(c1, c2, 1, 1))
                layers.append(Conv(c1, c2, 1, 1))
            else: # depth 2 ~
                layers.append(Conv(c2, c2, k, 1))
                layers.append(Conv(c2, c2, k, 1))
        # make layers sequential like yolo
        self.layers = nn.Sequential(*layers)
        # active index is used for forward
        self.act_idx = [idx for idx in range(depth * 2) if (idx % 2 == 1 or idx == 0)]


    def forward(self, x, d=None):
        outputs = []
        for i, m in enumerate(self.layers):
            if i == 0: # left output in depth 1
                outputs.append(m(x))
            else: # right outputs in depth 1 ~
                x = m(x)    # it is equivalent with f = [-1] in config(self.yaml)
                outputs.append(x)
                
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1][::-1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx[::-1]], dim=1)


    def get_active_net(self):
        raise NotImplementedError


class HeadELAN(ELAN):
    mode = 'Head'
    def __init__(self, c1, c2, k, depth):
        super(HeadELAN, self).__init__(c1, c2, k, depth)
        assert c1 % 2 == 0 and c2 % 2 == 0
        c_ = int(c2 / 2)
        
        layers = []
        # make layers according to depth
        for i in range(depth):
            if i == 0: # depth 1
                layers.append(Conv(c1, c2, 1, 1))
                layers.append(Conv(c1, c2, 1, 1))
            elif i == 1: # depth 2
                layers.append(Conv(c2, c_, k, 1))
            else: # depth 3 ~
                layers.append(Conv(c_, c_, k, 1))
        # make layers sequential like yolo
        self.layers = nn.Sequential(*layers)
        # active index is used for forward
        self.act_idx = [idx for idx in range(depth + 1)]
        
        
    def forward(self, x, d=None):
        outputs = []
        for i, m in enumerate(self.layers):
            if i == 0: # left output in depth 1
                outputs.append(m(x))
            else: # right outputs in depth 1 ~
                x = m(x)
                outputs.append(x)
                
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1][::-1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx[::-1]], dim=1)
    
    
    def get_active_net(self):
        raise NotImplementedError


class ELAN2(nn.Module):
    def __init__(self, c1, c2, k, depth, act):
        super(ELAN2, self).__init__()
        assert c1 % 2 == 0
        self.c2 = c2
        self.depth = depth
        self.act = act


class TinyELAN(ELAN2):
    mode = 'TinyELAN'
    def __init__(self, c1, c2, k, depth, act): # activation default : SILU() / inputs could be : nn.ReLU(), nn.LeakyReLU() ...
        super(TinyELAN, self).__init__(c1, c2, k, depth, act)
        assert c1 % 2 == 0

        layers = []
        # make layers according to depth
        for i in range(depth*2):
            if i in [0, 1]: # depth 1
                layers.append(Conv(c1, c2, 1, 1, act=act)) # Conv need activation
            else: # depth 2 ~
                layers.append(Conv(c2, c2, k, 1, act=act))
        # make layers sequential like yolo
        self.layers = nn.Sequential(*layers)
        # active index is used for forward
        self.act_idx = [idx for idx in range(depth*2)]


    def forward(self, x, d=None):
        outputs = []
        for i, m in enumerate(self.layers):
            if i == 0: # left output in depth 1
                outputs.append(m(x))
            else: # right outputs in depth 1 ~
                x = m(x)
                outputs.append(x)
                
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1][::-1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx[::-1]], dim=1)
    
    def get_active_net(self):
        raise NotImplementedError


class ELANBlock(nn.Module):
    def __init__(self, mode, layers, depth):
        super(ELANBlock, self).__init__()
        self.layers = layers
        if mode == 'BBone':
            self.act_idx = [idx for idx in range(depth * 2) if (idx % 2 == 1 or idx == 0)] # it is only for Bbone
        elif mode == 'Head':
            self.act_idx = [idx for idx in range(depth + 1)]
        else:
            raise ValueError
            
    def forward(self, x, d=None):
        outputs = []
        for i, m in enumerate(self.layers):
            if i == 0:
                outputs.append(m(x))
            else:
                x = m(x)
                outputs.append(x)
                
        if d is not None:
            return torch.cat([outputs[i] for i in self.act_idx[:d+1][::-1]], dim=1)
        return torch.cat([outputs[i] for i in self.act_idx[::-1]], dim=1)


# ======================== dynamic modules =====================================
# 1. DyConv: nn.Conv2d, F.conv2d, nn.BatchNorm2d, F.batch_norm, nn.SiLU, nn.Identity
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
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(MyConv2d, self).forward(x)
        else:
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        return super(MyConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)

        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        self.padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters,
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     dilation=self.dilation,
                     groups=1
                    )
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DyConv(nn.Module):
    # Dynamic Convolution for elastic channel size
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride
        super(DyConv, self).__init__()
        self.conv = DynamicConv2d(c1, c2, k, s) # auto same padding
        self.bn = DynamicBatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def fuse_conv_and_bn(self):
        conv = self.conv.conv
        bn = self.bn.bn
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        if self.bn is not None:
            del self.bn

        return fusedconv


class DyConvBlock(nn.Module):
    #  DyConvBlock for subnet
    def __init__(self, conv, bn, act, in_channels):
        super(DyConvBlock, self).__init__()
        c1 = in_channels
        c2 = conv.max_out_channels
        k  = conv.kernel_size
        p  = conv.padding
        s  = conv.stride
        b  = False

        filters = conv.get_active_filter(c2, c1)
        self.conv = DynamicConv2d(c1, c2, k, s)
        with torch.no_grad():
            self.conv.conv.weight.copy_(filters)

        self.bn = bn
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def fuse_conv_and_bn(self):
        conv = self.conv.conv
        bn = self.bn.bn
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)
        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


# ======================== rep modules =========================================
# 1. RepConv: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity, F.pad
# 2. RepNCSPELAN4: nn.Conv2d, nn.BatchNorm, nn.SiLU, nn.Identity, F.pad
class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True).requires_grad_(False)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode).requires_grad_(False)

        conv.weight = torch.nn.Parameter(weights).requires_grad_(False)
        conv.bias = torch.nn.Parameter(bias).requires_grad_(False)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        # print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False).requires_grad_(False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) ).requires_grad_(False)
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) ).requires_grad_(False)

        # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded).requires_grad_(False)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded).requires_grad_(False)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# ======================== detect modules ======================================
# 1. Detect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 2. IDetect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 3. IAuxDetect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 4. DDetect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 5. DualDDetect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
# 6. TripleDDetect: nn.Conv2d, nn.BatchNorm2d, nn.SiLU, nn.Identity
class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # type: ignore[attr-defined]
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # export torchscript or onnx w/o nms
    end2end = False # export onnx w/nms
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference or no export
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference or no export
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                # if not torch.onnx.is_in_onnx_export():
                #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # else:
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)


class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])
        #y = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return y if self.export else (y, d2)
        #y1 = torch.cat((dbox, cls.sigmoid()), 1)
        #y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
        #return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), 
                          nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), 
                          nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4), 
                          nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        #y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        #return y if self.export else (y, [d1, d2, d3])
        y = torch.cat((dbox3, cls3.sigmoid()), 1)
        return y if self.export else (y, d3)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


# ======================== model creation ======================================
# 1. ClassifyModel
# 2. DetectModel
# 3. NASModel
import yaml
import json
from pathlib import Path
import numpy as np

class ClassifyModel(nn.Module):
    def __init__(self, cfg='basemodel.yaml', ch=1, nc=None):
        super(ClassifyModel, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        # channel (grey = 1, rgb = 3)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        # number of class
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value

        # image size
        imgsz = self.yaml['imgsz']

        # define model
        self.model, self.save = parse_cls_model(deepcopy(self.yaml), ch=[ch])

        # default class name
        self.names = [str(i) for i in range(self.yaml['nc'])]

        # init weights
        self.initialize_weights()

        # model info
        self.info(imgsz, verbose=False)

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=1)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if profile:
                o = thop.profile(m, inputs=x, verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x)
                t = time_synchronized()
                for _ in range(10):
                    m(x)
                dt.append((time_synchronized() - t) * 100)
                logger.info('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def initialize_weights(self, zero_init_residual=True):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.model.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, (BasicBlock, cBasicBlock)) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def fuse(self):  # fuse cv+bn or cv+implict ops into one cv op
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block() # update conv, remove bn-modules, and update forward
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()  # update conv, remove bn-modules, and update forward
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        # self.info()
        imgsz = self.yaml['imgsz']
        self.summary(imgsz, verbose=False)
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


class DetectModel(nn.Module):
    """ Create YOLO-based Detector NN-Model

    Args:
    -----------
    cfg: str
        path to model configuration yaml file
    ch: int
        number of input channels
    nc: int
        number of classes
    anchors: 2d-list or int
        anchors : np * 3(anchros/floor) * 2(width & height)

    Attributes:
    -----------
    traced: bool
        true if this model is traced by torchscript, onnx
    yaml: dict
        dict loaded from model configuration yaml file
    yaml_file: str
        path to model configuration yaml file
    model: nn.Sequential
        a sequence of nn.Modules
    save: list of int
        a save list of output channels at nn.Modules
    names: list of str
        class names (labelmaps)
    stride: int
        yolo-specific grid size

    """
    def __init__(self, cfg='bestmodel.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(DetectModel, self).__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            self.yaml['anchors'] = round(anchors)  # override yaml value
        print('_'*100)
        self.model, self.save = parse_det_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        print('_'*100)
        print(f"skip connection list: {self.save}")
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, IDetect)):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        if isinstance(m, IAuxDetect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once
        if isinstance(m, DDetect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        if isinstance(m, (DualDDetect, TripleDDetect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info(verbose=True)

    def forward(self, x):
            return self.forward_once(x)

    def forward_once(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


class NASModel(DetectModel):
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
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                    break

            if profile:
                import thop
                c = isinstance(m, (Detect, IDetect, IAuxDetect))
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
            logger.info('%.1fms total' % sum(dt))
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


class ModelLoader(nn.Module):
    def __init__(self, weights='bestmodel.pt', cfg='bestmodel.yaml', task='detection', device=torch.device('cpu'), fuse=True):
        super().__init__()
        print('-'*50+'ModelLoader'+'-'*50)
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, onnx_end2end, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        # fp16 &= pt or jit or onnx # FP16
        fp16 = False
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if pt:  # PyTorch
            print(f'Loading {w} for PyTorch inference...')
            model = load_weights(cfg, weights if isinstance(weights, list) else w, task, map_location=device, fused=fuse)
            
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            # model.half() if fp16 else model.float()
            model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            print(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            # model.half() if fp16 else model.float()
            model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif onnx:  # ONNX Runtime
            logger.info(f'Loading {w} for ONNX Runtime inference...')
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')
        model.eval()

        self.__dict__.update(locals())  # assign all variables to self

        # class names
        # if 'names' not in locals():
        #     self.names = {i: f'class{i}' for i in range(999)}

    def forward(self, im, augment=False, visualize=False):
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            print('model forward: fp16')
            im = im.half()  # to FP16
        # if self.nhwc:
        #     print('model forward: nhwc')
        #     im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            print('model forward: pt model')
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            print('model forward: torchscript model')
            y = self.model(im)
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from tango.main.export import export_formats
        sf = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = False
        return types + [triton]

    @staticmethod
    def _load_metadata(filepath=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if filepath.exists():
            with open(filepath, 'r') as f:
                d = yaml.safe_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None

# ======================== utilities ===========================================
import math
import time
from copy import deepcopy

def parse_cls_model(d, ch):  # model_dict, input_channels(1 or 3)
    nc = d['nc']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [   nn.Conv2d,
                    Conv,
                    RepConv,
                    DownC,
                    SPPCSPC,
                    cBasicBlock,
                    BasicBlock,
                    Bottleneck,
                    nn.Linear   ]:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
                c2 = make_divisible(c2, 8)
            args = [c1, c2, *args[1:]]
            if m in [   DownC,
                        SPPCSPC  ]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is ReOrg:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '').split('.')[-1]  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def parse_det_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [   nn.Conv2d, 
                    Conv, 
                    DownC, 
                    SPPCSPC, 
                    SPPELAN, 
                    Bottleneck,
                    BasicBlock,
                    cBasicBlock,
                    DyConv, 
                    AConv, 
                    ADown, 
                    ELAN1, 
                    RepNCSPELAN4    ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is BBoneELAN:
            c1, c2 = ch[f], int(args[0]*(args[-1]+1))
            args = [c1, *args]
        elif m is HeadELAN:
            c1, c2 = ch[f], int((args[0]*2) + (args[0]/2 * (args[-1]-1)))
            args = [c1, *args]
        elif m is TinyELAN:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [ Detect, 
                    IDetect, 
                    IAuxDetect, 
                    DDetect, 
                    DualDDetect, 
                    TripleDDetect ]:
            args.append([ch[x] for x in f])
            if m in [Detect, IDetect, IAuxDetect]:
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '').split('.')[-1]  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%38s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def load_weights(cfg, weights, task, map_location=None, fused=True):
    if task == 'detection':
        print(f'making detection model from {cfg}...')
        model = DetectModel(cfg=cfg)
    elif task == 'classification':
        model = ClassifyModel(cfg=cfg)
    print(weights)
    ckpt = torch.load(weights, map_location=map_location, weights_only=True)  # load
    print('#'*100)
    # Model compatibility updates
    # if not hasattr(ckpt, 'stride'):
    #     ckpt.stride = torch.tensor([32.])

    if fused:
        csd = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().state_dict()
    else:
        csd = ckpt['ema' if ckpt.get('ema') else 'model'].float().state_dict()

    model.load_state_dict(csd, strict=True)

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    return model

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception) as e:
        print(e)
        fs = ''

    print('='*100)
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
    print('='*100)

def check_suffix(file='yolo.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()
            if len(s):
                if not s in suffix:
                    print(f"{msg}{f} acceptable suffix is {suffix}")
