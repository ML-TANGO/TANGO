import torch
import torch.nn as nn

from tango.common.models.common import Conv, Concat
from tango.common.models.dynamic_op import DynamicConv2d, DynamicBatchNorm2d

# ELAN base module
class ELAN(nn.Module):
    def __init__(self, c1, c2, k, depth):
        super(ELAN, self).__init__()
        assert c1 % 2 == 0
        self.c2 = c2
        self.depth = depth


# ELANBlock in real
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


# ELANBlock for Backbone
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
    

# ELANBlock for Head
# there are differences about cardinality(path) and channel size
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


# DyConvBlock for subnet
class DyConvBlock(nn.Module):
    # Dynamic Convolution for elastic channel size
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

# Dynamic Convolution for elastic channel size
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


#===============================================================================
#=============================== YOLOv7-tiny ===================================
#===============================================================================

class ELAN2(nn.Module):
    def __init__(self, c1, c2, k, depth, act):
        super(ELAN2, self).__init__()
        assert c1 % 2 == 0
        self.c2 = c2
        self.depth = depth
        self.act = act

# backbone ELAN and head ELAN are the same in tiny version 
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
    
# Dynamic Convolution for elastic channel size
class TinyDyConv(nn.Module):
    # Dynamic Convolution for elastic channel size
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride
        super(TinyDyConv, self).__init__()
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
