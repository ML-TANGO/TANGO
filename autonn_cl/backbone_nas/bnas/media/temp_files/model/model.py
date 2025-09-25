from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class BestModel(nn.Module):
    '''model sampling
    param best_arch: it represents the architecture of final model
                     it was be included in neural_net_info.yaml
    '''
    def __init__(self, best_arch):
        super().__init__()
        head = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
        self.head = head.model[10:]
        self.supernet = BestBackbone(best_arch)
        self.supernet.set_backbone_fpn(returned_layers=[1, 2, 3])

    def forward(self, im):
        # return outputs, outputs.keys()
        o, k = self.supernet.forward_fpn(im) 
        multiScaleFs = [o[k[0]], o[k[1]], o[k[2]]]
        x = o[k[2]]

        for m in self.head:
            if m.f != -1:
                if m.f[1]>7:
                    x = [x if j == -1 else multiScaleFs[j-7] for j in m.f]
                elif m.f[1]==6:
                    x = [x, multiScaleFs[1]]
                elif m.f[1]==4:
                    x = [x, multiScaleFs[0]]

            if isinstance(m, Conv):
                in_C = m.conv.in_channels
                pad = torch.zeros(x.shape[0], in_C, x.shape[-2], x.shape[-1]).cuda()
                pad[:, :x.shape[1], ...] = x
                x = pad
            elif isinstance(m, C3):
                in_C = m.cv1.conv.in_channels
                pad = torch.zeros(x.shape[0], in_C, x.shape[-2], x.shape[-1]).cuda()
                pad[:, :x.shape[1], ...] = x
                x = pad
            x = m(x)
            multiScaleFs.append(x)
        return x

class BestBackbone(nn.Module):
    def __init__(self,
                arch,
                ):
        super(BestBackbone, self).__init__()
        
        b_ks = arch['ks']
        b_expand_ratio = arch['e']
        self.b_depth = arch['d']

        # first conv layer
        self.first_conv = ConvLayer(
            3, 16, kernel_size=3, stride=2
        )
        first_block_conv = MBConvLayer(
            in_channels=16,
            out_channels=16,
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(16)
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        block_index = 1
        feature_dim = 16

        width_list = [24, 40, 80, 112, 160]
        n_block_list = [4, 4, 4, 4, 4]
        stride_stages = [2, 2, 2, 1, 2]
        act_stages = ["relu", "relu",
                        "h_swish", "h_swish", "h_swish"]
        se_stages = [False, True, False, True, True]

        ldx = 0
        bdx = 0

        for width, n_block, s, act_func, use_se \
            in zip(width_list,
                   n_block_list,
                   stride_stages,
                   act_stages,
                   se_stages):
            self.block_group_info.append(
                [block_index + i for i in range(n_block)])
            block_index += n_block

            output_channel = width
            for i in range(n_block):
                if self.b_depth[bdx] <= i:
                    ldx += 1
                    continue
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = MBConvLayer(
                    in_channels = feature_dim,
                    out_channels = output_channel,
                    kernel_size = b_ks[ldx],
                    expand_ratio = b_expand_ratio[ldx],
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                if ((stride == 1 and
                     feature_dim == output_channel)):
                    shortcut = IdentityLayer(feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv,
                                            shortcut))
                feature_dim = output_channel
                ldx += 1
            bdx += 1

        self.blocks = nn.ModuleList(blocks)

        # final expand layer, feature mix layer & classifier
        self.final_expand_layer = ConvLayer(
            feature_dim, 960,
            kernel_size=1,
        )
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=True)
        self.feature_mix_layer = ConvLayer(
            960,
            1280,
            kernel_size=1,
            use_bn=False,
        )
        self.classifier = LinearLayer(
            1280, 1000)

    def set_backbone_fpn(self, returned_layers):
        """
            Specifying features for FPN & return in_channels_list
        """
        # [1, 5, 9, 17] except first conv it can be changed by depth
        stage_indices = [i for i, block in enumerate(
            self.blocks) if block.conv.stride > 1]
        num_stages = len(stage_indices)

        # freeze backbone
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(False)

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        self.return_layers = [stage_indices[i] for i in returned_layers]

        in_channels_list = [self.blocks[stage_indices[i]].conv.out_channels
                            if i != 0
                            else self.blocks[stage_indices[i]].out_channels
                            for i
                            in returned_layers]

        return in_channels_list
    
    def forward_fpn(self, x):
        '''
        compute fpn
        '''
        # FPN outputs
        outputs = OrderedDict()

        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        for idx in range(sum(self.b_depth)):
            x = self.blocks[idx](x)
            if ((self.return_layers is not None and
                idx in self.return_layers)):
                outputs[str(idx)] = x
            elif (self.return_layers is None
                and idx == self.block_group_info[-1][-1]):
                outputs[str(idx)] = x
        return outputs, list(outputs.keys())
        
class MyGlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(MyGlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return x.mean(3, keepdim=self.keep_dim).mean(2, keepdim=self.keep_dim)

    def __repr__(self):
        return "MyGlobalAvgPool2d(keep_dim=%s)" % self.keep_dim

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(LinearLayer, self).__init__()

        """ modules """
        self.linear = nn.Linear(in_features, out_features, True)

    def forward(self, x):
        return self.linear(x)

class IdentityLayer(nn.Module):
    def __init__(
        self,
        in_channels,
    ):
        super(IdentityLayer, self).__init__()
        # self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        return x # self.bn(x)

class ResidualBlock(nn.Module):
    def __init__(self, conv, shortcut):
        super(ResidualBlock, self).__init__()
        self.conv = conv
        self.shortcut = shortcut
    def forward(self, x):
        if self.shortcut is None:
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
        return res

def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    else:
        raise ValueError("do not support: %s" % act_func)

class MBConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        act_func="relu",
        use_se= False,
    ):
        super(MBConvLayer, self).__init__()

        self.stride = stride
        self.out_channels = out_channels
    
        feature_dim = round(in_channels * expand_ratio)

        if expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_channels, feature_dim, 1, 1, 0, bias=False
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(feature_dim)),
                        ("act", build_activation(act_func, inplace=True)),
                    ]
                )
            )

        depth_conv_modules = [
            (
                "conv",
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=feature_dim,
                    bias=False,
                ),
            ),
            ("bn", nn.BatchNorm2d(feature_dim)),
            ("act", build_activation(act_func, inplace=True)),
        ]
        if use_se:
            depth_conv_modules.append(("se", SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))
        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
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

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

def make_divisible(v, divisor, min_val=None):
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SEModule(nn.Module):
    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        reduction = 4
        num_mid = make_divisible(
            channel // reduction, divisor=8
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("reduce", nn.Conv2d(channel, num_mid, 1, 1, 0, bias=True)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("expand", nn.Conv2d(num_mid, channel, 1, 1, 0, bias=True)),
                    ("h_sigmoid", Hsigmoid(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        use_bn=True,
    ):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.act = Hswish(inplace=use_bn)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x

def autopad(k, p=None):  # kernel, padding
    '''
    autopad
    '''
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    '''
    Standard convolution
    ch_in, ch_out, kernel, stride, padding, groups
    '''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        '''
        forward
        '''
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        '''
        forward fuse
        '''
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    '''
    Standard bottleneck
    ch_in, ch_out, shortcut, groups, expansion
    '''
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, _c, 1, 1)
        self.cv2 = Conv(_c, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, _x):
        '''
        forward
        '''
        return (_x + self.cv2(self.cv1(_x))
                if self.add else self.cv2(self.cv1(_x)))

class C3(nn.Module):
    '''
    CSP Bottleneck with 3 convolutions
    ch_in, ch_out, number, shortcut, groups, expansion
    '''
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c, 1, 1)
        self.cv2 = Conv(c1, c, 1, 1)
        self.cv3 = Conv(2 * c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c, c, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        '''
        forward
        '''
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



if __name__ == "__main__":

    best_arch = {'ks': [3, 5, 7, 3, 5, 7, 5, 3, 5, 3, 3, 5, 5, 7, 3, 5, 3, 3, 3, 7], 'e': [3, 3, 3, 6, 3, 3, 4, 4, 3, 4, 3, 6, 4, 3, 4, 6, 4, 3, 6, 3], 'd': [2, 3, 3, 2, 2], 'r': [224]}
    model = BestModel(best_arch) 

    path = "shared/common/user_id/project_id/best.pt" 
    model.load_state_dict(torch.load(path))