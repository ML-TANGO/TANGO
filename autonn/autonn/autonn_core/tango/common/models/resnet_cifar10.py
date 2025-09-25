"""
ResNet CIFAR10 Model Definition
"""
import yaml
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging

from tango.common.models.common import *
from tango.common.models.experimental import *
from tango.utils.general import make_divisible #, check_file, set_logging
from tango.utils.torch_utils import (   fuse_conv_and_bn,
                                        model_summary,
                                        scale_img,
                                    )
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, Union
# import torchsummary

logger = logging.getLogger(__name__)

#=========================torchvision_models_resnet=============================
# imgsz = 224 x 224
# 4-element layer repetition [a, b, c, d]
# resnet-N : N = (a+b+c+d)*2+2
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


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
#=========================torchvision_models_resnet=============================


#=========================resnet-cifar10========================================
# imgsz = 256 x 256
# 3-element layer repetition [a, b, c]
# renset-N : N = (a+b+c)*2+2
class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


def conv3x3c(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class cResNet(nn.Module):
    @classmethod
    def load_config(cls, yaml_path: str) -> "ResNet":
        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return cls(BasicBlock, config["layers"], 2)

    def __init__(
        self,
        block,
        layers,
        num_classes: int = 10,
    ) -> None:
        super(cResNet, self).__init__()
        self.inplanes = 16

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet20(num_classes):
    model = cResNet(BasicBlock, [3, 3, 3], num_classes)
    return model


def resnet32(num_classes):
    model = cResNet(cBasicBlock, [5, 5, 5], num_classes)
    return model


def resnet44(num_classes):
    model = cResNet(cBasicBlock, [7, 7, 7], num_classes)
    return model


def resnet56(num_classes):
    model = cResNet(cBasicBlock, [9, 9, 9], num_classes)
    return model


def resnet110(num_classes):
    model = cResNet(cBasicBlock, [18, 18, 18], num_classes)
    return model


def resnet152(num_classes):
    model = cResNet(cBasicBlock, [25, 25, 25], num_classes)
    return model


def resnet200(num_classes):
    model = ResNet(cBasicBlock, [33, 33, 33], num_classes)
    return model

#=========================resnet-cifar10========================================


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
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value

        # image size
        imgsz = self.yaml['imgsz']

        # define model
        self.model, self.save, self.nodes_info = parse_model(deepcopy(self.yaml), ch=[ch])

        # default class name
        self.names = [str(i) for i in range(self.yaml['nc'])]

        # init weights
        self.initialize_weights()

        # model info
        self.briefs = self.summary(imgsz, verbose=False)

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
                import thop
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

    def summary(self, img_size=640, verbose=False):
        return model_summary(self, img_size, verbose)


def parse_model(d, ch):  # model_dict, input_channels(1 or 3)
    # Parse a classification model.yaml dictionay
    logger.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nodes_info = {}
    nc = d['nc']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # ==========================================================================
    # Shape tracking initialization
    # ==========================================================================
    imgsz = d.get('imgsz', 640)
    shapes = [(ch[0], imgsz, imgsz)]

    # ==========================================================================
    # Layer parsing loop
    # ==========================================================================
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        # ----------------------------------------------------------------------
        # Input shape
        # ----------------------------------------------------------------------
        in_shapes = [shapes[x] for x in (f if isinstance(f, list) else [f])] # [(C,H,W), ...]

        # ----------------------------------------------------------------------
        # Channel propagation (determine c2 and ajust agrs)
        # ----------------------------------------------------------------------
        if m in [
            nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv,
            RepConv, RepConv_OREPA, DownC, SPP, SPPF, SPPCSPC, GhostSPPCSPC,
            MixConv2d, Focus, Stem, GhostStem, CrossConv, cBasicBlock,
            Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
            RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
            Res, ResCSPA, ResCSPB, ResCSPC,
            RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
            ResX, ResXCSPA, ResXCSPB, ResXCSPC,
            RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
            Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
            SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
            SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC,
        ]:
            c1, c2 = ch[f], make_divisible(args[0], 8)

            args = [c1, c2, *args[1:]]
            if m in [
                DownC, SPPCSPC, GhostSPPCSPC,
                BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                ResCSPA, ResCSPB, ResCSPC,
                RepResCSPA, RepResCSPB, RepResCSPC,
                ResXCSPA, ResXCSPB, ResXCSPC,
                RepResXCSPA, RepResXCSPB, RepResXCSPC,
                GhostCSPA, GhostCSPB, GhostCSPC,
                STCSPA, STCSPB, STCSPC,
                ST2CSPA, ST2CSPB, ST2CSPC
            ]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Flatten:
            # Normalize args to PyTorch signature: (start_dim=1, end_dim=-1)
            # Many YAMLs mistakenly put tuples/dicts here; ignore them.
            if len(args) == 0:
                args = []  # use defaults
            elif len(args) == 1:
                logger.info('1'*100)
                # If someone put a tuple or dict, ignore and use defaults
                if isinstance(args[0], (tuple, list, dict)) or args[0] is None:
                    logger.warning("Ignoring invalid Flatten arg; using defaults (start_dim=1, end_dim=-1)")
                    args = []
                else:
                    # keep single int as start_dim
                    args = [int(args[0])]
            else:
                logger.info('2'*100)
                # keep first two as ints (start_dim, end_dim), drop the rest
                args = [int(args[0]), int(args[1])]

            c2 = ch[f]  # channels unchanged
        elif m is nn.Linear:
            if len(in_shapes) != 1:
                logger.warning("Linear expects a single input tensor")
            C, H, W = in_shapes[0]
            in_features = C * H * W
            out_features = int(args[0])
            args = [in_features, out_features, *args[1:]]
            c2 = out_features
        else:
            c2 = ch[f]

        # ----------------------------------------------------------------------
        # Module creation
        # ----------------------------------------------------------------------
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '').split('.')[-1]  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info(f'{i:>3}{str(f):>18}{n:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print

        nodes_info[f"{i:02d}"] = {
            "from": f,
            "repeat": n,
            "params": np,
            "module": t,
            "arguments": str(args),
        }
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        # ----------------------------------------------------------------------
        # Shape propagation
        # ----------------------------------------------------------------------
        multi_input = len(in_shapes) > 1
        if multi_input:
            cur_shape = propagate_shape(m, args, in_shapes)
        else:
            cur_shape = in_shapes[0]
            n_eff = n if isinstance(n, int) else 1
            for _ in range(n_eff):
                cur_shape = propagate_shape(m, args, [cur_shape])

        shapes.append(cur_shape)

        # ----------------------------------------------------------------------
        # Channel update
        # ----------------------------------------------------------------------
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save), nodes_info


def propagate_shape(m, args, in_shapes):
    """
    Shape inference for a single application of module class `m` with constructor args `args`.
    Works before module instantiation. Returns (C_out, H_out, W_out) when determinable,
    otherwise returns a conservative pass-through of the first input.
    Assumptions:
      - For Conv-like layers, args are normalized to [inC, outC, k, s, p, d, ...]
      - For Linear, args are normalized to [in_features, out_features, ...]
      - For Flatten, no args needed.
    """
    # ---- helpers ----
    def _conv_out_len(l_in, k, s, p, d=1):
        return ((l_in + 2*p - d*(k-1) - 1) // s + 1)

    def _as_hw_tuple(x, default):
        # normalize int/tuple to (h,w)
        if isinstance(x, (tuple, list)):
            return int(x[0]), int(x[1])
        return int(x), int(x)

    def _same_hw(hws):
        h0, w0 = hws[0]
        for (h, w) in hws[1:]:
            if h != h0 or w != w0:
                raise ValueError(f"Multi-input H,W mismatch: {hws}")
        return h0, w0

    single = (len(in_shapes) == 1)
    first = in_shapes[0]

    # ---- multi-input ops ----
    if m in (Concat, Chuncat):
        # H,W must match; C sums
        H, W = _same_hw([(h, w) for (_, h, w) in in_shapes])
        C = sum(C for (C, _, _) in in_shapes)
        return (C, H, W)

    if m is Shortcut:
        # elementwise add: all (C,H,W) identical
        C0, H0, W0 = in_shapes[0]
        for (C, H, W) in in_shapes[1:]:
            if (C, H, W) != (C0, H0, W0):
                raise ValueError(f"Shortcut shape mismatch: {in_shapes}")
        return (C0, H0, W0)

    # ---- single-input ops below ----
    if not single:
        # Unknown multi-input type → conservative: pass-through first
        return first

    C_in, H_in, W_in = first

    # Foldcut
    if m is Foldcut:
        return (C_in // 2, H_in, W_in)

    # ReOrg / Focus
    if m in (ReOrg, Focus):
        return (C_in * 4, H_in // 2, W_in // 2)

    # Contract / Expand
    if m is Contract:
        s = args[0]
        return (C_in * (s**2), H_in // s, W_in // s)

    if m is Expand:
        s = args[0]
        return (C_in // (s**2), H_in * s, W_in * s)

    # Flatten
    if m is nn.Flatten:
        return (C_in * H_in * W_in, 1, 1)

    # Linear (args already normalized)
    if m is nn.Linear:
        out_features = args[1] if len(args) > 1 else args[0]
        return (int(out_features), 1, 1)

    # BatchNorm2d
    if m is nn.BatchNorm2d:
        return (C_in, H_in, W_in)

    # MaxPool2d / AvgPool2d  (args: k, s=None, p=0, ...)
    if m in (nn.MaxPool2d, nn.AvgPool2d):
        k = args[0] if len(args) > 0 else 2
        s = args[1] if len(args) > 1 and args[1] is not None else k
        p = args[2] if len(args) > 2 else 0
        kh, kw = _as_hw_tuple(k, 2)
        sh, sw = _as_hw_tuple(s, kh)
        ph, pw = _as_hw_tuple(p, 0)
        H_out = _conv_out_len(H_in, kh, sh, ph, 1)
        W_out = _conv_out_len(W_in, kw, sw, pw, 1)
        return (C_in, H_out, W_out)

    # AdaptiveAvgPool2d (args[0] can be int or (H_out, W_out))
    if m is nn.AdaptiveAvgPool2d:
        out_sz = args[0] if len(args) > 0 else 1
        if isinstance(out_sz, int):
            return (C_in, int(out_sz), int(out_sz))
        return (C_in, int(out_sz[0]), int(out_sz[1]))

    # Upsample (constructor may pass size= or scale_factor=. Here we read from args if present.)
    if m is nn.Upsample:
        # best-effort: accept either kw-style dict or positional list
        # common: kwargs dict as last arg or only arg
        size = None
        scale_factor = None
        # try kwargs dict at the end
        if args and isinstance(args[-1], dict):
            size = args[-1].get('size')
            scale_factor = args[-1].get('scale_factor')
        # or positional (rare in yaml)
        if size is None and len(args) >= 1 and isinstance(args[0], (tuple, list, int)):
            # heuristics: treat first positional as size if provided
            size = args[0] if not isinstance(args[0], (float,)) else None
        if scale_factor is None and len(args) >= 1 and isinstance(args[0], (float, tuple, list)):
            scale_factor = args[0] if isinstance(args[0], (float, tuple, list)) else None

        if size is not None:
            if isinstance(size, (tuple, list)):
                return (C_in, int(size[0]), int(size[1]))
            return (C_in, int(size), int(size))
        if scale_factor is None:
            scale_factor = 2
        if isinstance(scale_factor, (tuple, list)):
            sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh = sw = float(scale_factor)
        return (C_in, int(round(H_in * sh)), int(round(W_in * sw)))

    # ConvTranspose2d (args: inC, outC, k, s, p, out_pad, groups, bias, d)
    if m is nn.ConvTranspose2d:
        k = args[2] if len(args) > 2 else 2
        s = args[3] if len(args) > 3 else 2
        p = args[4] if len(args) > 4 else 0
        op = args[5] if len(args) > 5 else 0
        d = args[8] if len(args) > 8 else 1
        kh, kw = _as_hw_tuple(k, 2)
        sh, sw = _as_hw_tuple(s, 2)
        ph, pw = _as_hw_tuple(p, 0)
        oph, opw = _as_hw_tuple(op, 0)
        # (in - 1)*s - 2p + d*(k-1) + out_pad + 1
        H_out = (H_in - 1) * sh - 2 * ph + d * (kh - 1) + oph + 1
        W_out = (W_in - 1) * sw - 2 * pw + d * (kw - 1) + opw + 1
        # C_out은 outC(= args[1])이지만, 여기서는 외부에서 채널 관리 가능
        C_out = int(args[1]) if len(args) > 1 else C_in
        return (C_out, H_out, W_out)

    # Conv-like / composite blocks (treat as single conv step):
    # Expect args like [inC, outC, k, s, p, d, ...] after your channel-prop normalization.
    if m in (nn.Conv2d, Conv, AConv, DWConv, Bottleneck, SPP, SPPF, SPPCSPC, ADown, ELAN1, RepNCSPELAN4, SPPELAN,
             DownC, Stem, CrossConv, GhostConv, RobustConv, RobustConv2, MixConv2d,
             RepConv, RepConv_OREPA, GhostStem,
             cBasicBlock, Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
             RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
             Res, ResCSPA, ResCSPB, ResCSPC, RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
             ResX, ResXCSPA, ResXCSPB, ResXCSPC, RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
             Ghost, GhostCSPA, GhostCSPB, GhostCSPC):
        C_out = int(args[1]) if len(args) > 1 else C_in
        k = args[2] if len(args) > 2 else 1
        s = args[3] if len(args) > 3 else 1
        p = args[4] if len(args) > 4 else (k // 2 if isinstance(k, int) else 0)
        d = args[5] if len(args) > 5 else 1
        kh, kw = _as_hw_tuple(k, 1)
        sh, sw = _as_hw_tuple(s, 1)
        ph, pw = _as_hw_tuple(p, kh // 2)
        dh, dw = _as_hw_tuple(d, 1)
        H_out = _conv_out_len(H_in, kh, sh, ph, dh)
        W_out = _conv_out_len(W_in, kw, sw, pw, dw)
        return (C_out, H_out, W_out)

    # Fallback: unknown layer → pass-through (safe default)
    return (C_in, H_in, W_in)