import math
from copy import deepcopy
from pathlib import Path
from pprint import pprint

# import nni.retiarii.nn.pytorch as nn
import torch
import torch.nn as nn

import yaml
# from neck import Neck
from backbone import Backbone
from ops import (Conv, CBL, CBM, CBR, CBR6, CBS, CSP, DBR6, MB, SPP,
                 Focus, Bottleneck, Concat, ConcatForNas, ConcatSPP,
                 Detect, NMS)
from yolov5_utils.autoanchor import check_anchor_order
from yolov5_utils.general import LOGGER
from yolov5_utils.torch_utils import (fuse_conv_and_bn, model_info,
                                      scale_img, time_sync,
                                      initialize_weights)
# from models.common import C3


class SearchNeck(nn.Module):
    """ Create neural networks from search space for Neck-NAS

    Parameters:
    -----------
    backbone_yaml: str
        path to backbone yaml file
    np: int
        number of pyramid-floors
    nc: int
        number of classes
    anchors: 2d-list
        anchors : np * 3(anchros/floor) * 2(width & height)

    Attributes:
    -----------
    model: nn.Sequential
        a sequence of nn.modules
    save: list
        indice of jumping points
    stride: list of int
        grid strides
    """
    def __init__(self, backbone_yaml, np=3, nc=80, anchors=[]):
        super(SearchNeck, self).__init__()
        # backbone
        with open(backbone_yaml, encoding='utf-8', errors='ignore') as f:
            backbone_dict = yaml.safe_load(f)
        bridge_id, bridge_ch = \
            parse_backbone(deepcopy(backbone_dict), np)
        backbone_len = len(backbone_dict['backbone'])
        # for id, ch in zip(bridge_id, bridge_ch):
        #     print(f'{id}: channels={ch}')

        # neck & head
        neck_yaml = 'autonn/yaml/neck.yaml'
        with open(neck_yaml, encoding='utf-8', errors='ignore') as f:
            neck_dict = yaml.safe_load(f)
        deepcopy_neck_dict = deepcopy(neck_dict)
        parse_neck(bridge_id, bridge_ch, np, backbone_len,
                   deepcopy_neck_dict, nc, anchors)

        # backbone + neck + head
        backbone_dict.update(deepcopy_neck_dict)
        # with open('autonn/yaml/yolov5l_p5f.yaml',
        #           encoding='ascii', errors='ignore') as f:
        #     backbone_dict = yaml.safe_load(f)
        fullarch_dict = deepcopy(backbone_dict)

        # define model
        self.model, self.save = build_model(fullarch_dict)
        self.names = [str(i) for i in range(nc)]
        self.inplace = fullarch_dict.get('inplace', True)

        # build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            ch = 3  # RGB color image
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)  # divided by 8, 16, and 32
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())

        # initialize basic modules(conv2d, batchnorm2d, activation)
        initialize_weights(self)
        self.info(verbose=False)
        # for n, m in enumerate(list(self.model.modules())):
        #     print(f'{n}: {m}')
        #     print('-------')

    def forward(self, x, augment=False, profile=False):
        LOGGER.debug(f'forward model')
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # save
                # cv2.imwrite(
                #     'img%g.jpg'
                #     % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1]
                # )

                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self._forward_once(x, profile)

    def _forward_once(self, x, profile=False):
        LOGGER.debug(f'forward once')
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) \
                           else [x if j == -1 else y[j] for j in m.f]

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,),
                                     verbose=False)[0] / 1E9 * 2  # FLOPS
                except BaseException:
                    o = 0
                t = time_sync()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    print(f"{'time (ms)':>10s}{'GFLOPs':>10s}"
                          f"  {'params':>10s} {'module'}")
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        """ initialize biases into Detect(),
            cf is class frequency
            cf = torch.bincount(
                    torch.tensor(
                        np.concatenate(dataset.labels, 0)[:, 0]
                    ).long(),
                    minlength=nc
                ) + 1.
            * biases in Detect() =  a kind of embeddings for predition
        """
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.last_conv, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None \
                else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        """ fuse model Conv2d() + BatchNorm2d() layers """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            if (isinstance(m, (Conv, CBR, CBR6, DBR6, CBL, CBS, CBM))
                    and hasattr(m, 'bn')):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def info(self, verbose=False, img_size=640):
        """ print model information """
        model_info(self, verbose, img_size)

    def init_model(self, model_init='he_fout', init_div_groups=False):
        """ initialize weights and biases """
        eps = 1e-7
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n + eps))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n + eps))
                else:
                    raise NotImplementedError
            elif (isinstance(m, nn.BatchNorm2d)
                  or isinstance(m, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1) + eps)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()


class SearchSpaceWrap(nn.Module):
    def __init__(self, backbone_yaml, neck, head, neck_cfg, np, dataset,
                 anchors, nc, device, path_freezing=None):
        super().__init__()

        if True:
            backbone_module = Backbone(backbone_yaml)

            self.nc = nc
            self.anchors = anchors
            # self.ch_out = backbone_module.channels

            # neck
            self.neck_module = neck(deepcopy(backbone_module), neck_cfg, np=np,
                                    device=device, path_freezing=path_freezing)
            # self.ch_out += self.neck_module.channels
            # self.layers = self.neck_module.layers  # (tenace: not need)
            self.ch_out = self.neck_module.channels  # (tenace: bb + nk)
            # print(self.ch_out)

            # neck-to-head
            head_ch_in = []
            for i in self.neck_module.connect:
                head_ch_in.append(self.ch_out[i])
            # print(self.neck_module.connect)
            # print(head_ch_in)

            len_model = len(backbone_module.layers) \
                + len(self.neck_module.layers)

            # head
            head = Detect(nc=self.nc, anchors=self.anchors, ch_in=head_ch_in)
            # self.layers.append(head)  # (tenace comment: bug! duplicate)
            t = "Detect"
            params = sum(x.numel() for x in head.parameters())
            head.i, head.f, head.type, head.params = \
                len_model, self.neck_module.connect, t, params
            # print(f'{head.type} added: {len(list(head.modules()))}')
            # full architecture
            # self.model = nn.ModuleList([*backbone_module.layers]
            #                            + [*self.neck_module.layers]
            #                            + [head])
            self.model = nn.Sequential(*backbone_module.layers,
                                       *self.neck_module.layers, head)

            """ for debug """
            print('----------------------backbone-------------------------')
            for _, layer in enumerate(backbone_module.layers):
                print('[%2d] from:%20s %13s %10d'
                      % (layer.i, layer.f, layer.type, layer.params))
            print('------------------------neck---------------------------')
            for _, layer in enumerate(self.neck_module.layers):
                print('[%2d] from:%20s %13s %10d'
                      % (layer.i, layer.f, layer.type, layer.params))
            print('------------------------head---------------------------')
            layer = head
            print('[%2d] from:%20s %13s %10d'
                  % (layer.i, layer.f, layer.type, layer.params))
            print('-' * 55)
        else:
            # backbone
            backbone_dict = []
            with open(backbone_yaml, encoding='utf-8', errors='ignore') as f:
                backbone_dict = yaml.safe_load(f)
            bridge_id, bridge_ch = \
                parse_backbone(deepcopy(backbone_dict), np)
            backbone_len = len(backbone_dict['backbone'])

            with open(neck_cfg, encoding='utf-8', errors='ignore') as f:
                neck_dict = yaml.safe_load(f)
            parse_super_neck(
                bridge_id, bridge_ch, np, backbone_len,
                neck_dict, nc, anchors)
            backbone_dict.update(neck_dict)
            fullarch_dict = deepcopy(backbone_dict)
            # pprint(fullarch_dict)

            # define model
            self.model, self.save = build_model(fullarch_dict)

        # trick: comment if you do not want AMP
        # self.names = [str(i) for i in range(nc)]

        # build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 608  # 2x min stride
            ch = 3  # RGB color image
            m.stride = torch.tensor(
                [s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)  # divided by 8, 16, and 32
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())

        # initialize weights and biases
        initialize_weights(self)
        self.info(verbose=False)
        # for n, m in enumerate(list(self.model.modules())):
        #     print(f'{n}: {m}')
        #     print('-------')

    def _initialize_biases(self, cf=None):
        """ initialize biases into Detect(),
            cf is class frequency
            cf = torch.bincount(
                    torch.tensor(
                        np.concatenate(dataset.labels, 0)[:, 0]
                    ).long(),
                    minlength=nc
                ) + 1.
            * biases in Detect() =  a kind of embeddings for predition
        """
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.last_conv, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None \
                else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x, augment=False, profile=False):
        LOGGER.debug(f'forward model')
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # save
                # cv2.imwrite(
                #     'img%g.jpg'
                #     % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1]
                # )

                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self._forward_once(x, profile)

    def _forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        # (tenace comment: bug! self.layers are only neck layers)
        # for m in self.layers:
        for m in self.model:
            # print(m.type)
            if m.f != -1:  # if it is not from previous layer
                x = y[m.f] if isinstance(m.f, int) \
                    else [x if j == -1 else y[j] for j in m.f]

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,),
                                     verbose=False)[0] / 1E9 * 2  # FLOPS
                except BaseException:
                    o = 0
                t = time_sync()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    print(f"{'time (ms)':>10s}{'GFLOPs':>10s}"
                          f"  {'params':>10s} {'module'}")
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x)

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def fuse(self):
        """ fuse model Conv2d() + BatchNorm2d() layers """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
            if (isinstance(m, (Conv, CBR, CBR6, DBR6, CBL, CBS, CBM))
                    and hasattr(m, 'bn')):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def info(self, verbose=False, img_size=640):
        """ print model information """
        model_info(self, verbose, img_size)

    def init_model(self, model_init='he_fout', init_div_groups=False):
        """ initialize weights and biases """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif (isinstance(m, nn.BatchNorm2d)
                  or isinstance(m, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()


class ConcatEntirePathNeck(nn.Module):
    def __init__(self, backbone, neck_cfg='neck.yaml',
                 np=3, device='cpu', path_freezing=None):
        super().__init__()

        # Temp: read yaml file
        self.cfg_file = Path(neck_cfg).name
        with open(neck_cfg, encoding='ascii', errors='ignore') as f:
            self.cfg = yaml.safe_load(f)
        self.layers, self.channels, self.connect, self.return_list = \
            parse_neck_for_concat_nas(
                deepcopy(self.cfg), backbone, np, device=device,
                return_list=path_freezing
            )

    def forward(self, x):
        return self._forward_once(x)

    def _forward_once(self, x):
        y = []  # outputs
        for m in self.layers:
            if m.f != -1:  # if it is not from previous layer
                x = y[m.f] if isinstance(m.f, int) \
                    else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x)
        return x


def parse_neck_for_concat_nas(nd, bm, p, device, return_list=None):
    """ nd=neck_model_dictionary,
        bm=backbone_module,
        p=number_pyramid-floors """
    # layers from pre-built backbone
    backbone_layers = bm.layers
    si = len(backbone_layers)  # starting index of 'neck'

    # layers for neck
    layers = []

    connect_fr = []  # layers from pre-built backbone
    neck_channel = []  # neck channel just in case no 'neck-channel' in yaml
    for t in bm.connect:
        connect_fr.append(t[0])
        neck_channel.append(t[1])
    # connect_fr[-1] += 1  # (tenace comment: why? SPP belongs to Neck)
    neck_channel.reverse()  # order: [P3, P4, P5] -> [P5, P4, P3]
    connect_fr.reverse()    # order: [P3, P4, P5] -> [P5, P4, P3]
    connect_fr_sorted = sorted(connect_fr)

    # channels from pre-built backbone
    ch = bm.channels

    # size(height or width) from pre-built backbone
    sz = bm.size

    # reference size per pyramid-floor
    refer_sz = []
    for index in connect_fr_sorted:
        refer_sz.append(sz[index])

    # Search Stage OR Retrain Stage(path freezing)
    # path_freezing = True
    # if return_list is None:
    #     return_list = []
    #     path_freezing = False
    # else:
    #     temp_list = []
    #     for line in return_list:
    #         temp_list.append(torch.Tensor(line))
    #     return_list = temp_list
    path_freezing = False

    for i, (f, n, m, args) in enumerate(nd['neck']):
        """ from, number(repetition), module(layer), arguments """
        m = eval(m) if isinstance(m, str) else m  # nn module name
        for j, a in enumerate(args):
            try:
                # arguments (list)
                args[j] = eval(a) if isinstance(a, str) else a
            except BaseException:
                pass

        if m in [Conv, CBR, CBR6, CBL, CBS, CBM, Bottleneck, CSP, SPP]:
            ch_in, ch_out = ch[f], args[0]
            sz_out = sz_in = sz[f]
            # [ch_in, ch_out] if CSP else [ch_in, ch_out, kernel, stride]
            args = [ch_in, ch_out, *args[1:]]
            if m is CSP:
                # [ch_in, ch_out, repetition, shortcut]
                args.insert(2, n)
            elif m in [Bottleneck, SPP]:
                pass
            else:
                if args[3] == 2:
                    sz_out = sz_in * 2
        elif m is MB:
            ch_in, ch_out = ch[f], args[1]
            sz_out = sz_in = sz[f]
            # [ch_in, ch_multiple(t), ch_out(c), repetition(n), stride(s)]
            args.insert(0, ch[f])
            if args[4] == 2:
                sz_out = sz_in * 2
        elif m is nn.Upsample:
            ch_out = ch_in = ch[f]
            sz_out = sz_in = sz[f]
            if args[1] == 2:
                sz_out = sz_in // 2
        elif m is ConcatForNas:
            # need to cosider the difference between backbone and prelayer
            # ... between search and retrain
            # m = ConcatForNas
            if path_freezing:
                return_list_cp = deepcopy(return_list)
                for idx, x in enumerate(deepcopy(f)):
                    if x == "Backbone":
                        f.pop(idx)
                        refer_ch = [ch[index] for index in connect_fr]
                        cc = return_list_cp.pop(0)
                        if len(cc) != len(connect_fr):
                            for c_fr_idx, t in enumerate(connect_fr):
                                if idx + c_fr_idx == 0:
                                    f.append(t)
                                else:
                                    if cc[c_fr_idx-1] > 0:
                                        f.append(t)
                        else:
                            for c_fr_idx, t in enumerate(connect_fr):
                                if cc[c_fr_idx] > .5:
                                    f.append(t)
                        ch_res_rule = 0
                    elif x == "PreLayer":
                        f.pop(idx)
                        refer_ch = [ch[index+si] for index in connect_fr]
                        cc = return_list_cp.pop(0)
                        # for c_fr_idx, t in enumerate([3, 7]):
                        for c_fr_idx, t in enumerate(nd['fpn-to-pan']):
                            if cc[c_fr_idx] > 0. and t != i - 1:
                                f.append(t+si)
                        ch_res_rule = 1
            else:  # path initialize
                for idx, x in enumerate(deepcopy(f)):
                    if x == "Backbone":
                        f.pop(idx)
                        for c_fr_idx, t in enumerate(connect_fr):
                            # if t != len(ch):
                            if len(ch) == si + 1 and t == si - 1:
                                pass
                            else:
                                f.append(t)
                        ch_res_rule = 0
                    elif x == "PreLayer":
                        f.pop(idx)
                        # for c_fr_idx, t in enumerate([3, 7]):
                        for c_fr_idx, t in enumerate(nd['fpn-to-pan']):
                            if t != i - 1:
                                # prevent previous layer from duplicating
                                f.append(t+si)
                        ch_res_rule = 1

                # refer_ch = [ch[index] for index in f if index != -1]
                refer_ch = []
                sz_in = sz[f[0]]
                for index in f:
                    # if index != -1 and ch[index] not in refer_ch:
                    if ch[index] not in refer_ch:
                        refer_ch.append(ch[index])
                    # if sz_in != sz[index]:
                    #     print('Concat must have all the same size inputs')
                    #     print(f'{sz_in} is not the same as {sz[index]}')
                    print(f'input {index} : {sz[index]}')

            # print(ch_res_rule)
            # if ch_res_rule == 0:
            #     ch_out = sum([ch[-1] for _ in f])
            # elif ch_res_rule == 1:
            #     ch_out = sum([ch[f_idx] for f_idx in f])
            # (tenace comment) PANet is the same thing as FPN
            #                  UP/DW sizing is always with channel scalining
            ch_out = sum([ch[-1] for _ in f])
            sz_out = sz_in
        elif m is Concat:
            if isinstance(f, list):
                for j, fr in enumerate(f):
                    if isinstance(fr, str):
                        if fr == 'Backbone':
                            fr = connect_fr_sorted.pop(-2)
                            # fr = connect_fr_sorted[0]
                        elif fr == 'PreLayer':
                            fr = nd['fpn-to-pan'].pop(-2) + si
                        else:
                            raise NotImplemented
                        f[j] = fr
                    else:
                        f[j] = fr + si if fr != -1 else fr
            else:
                f = f + si if f != -1 else f
            ch_out = sum([ch[x] for x in f])
            sz_in = sz[f[0]]
            for index in f:
                # if sz_in != sz[index]:
                #     print('Concat must have all the same size inputs')
                #     print(f'{sz_in} is not the same as {sz[index]}')
                print(f'input {index} : {sz[index]}')
        elif m is ConcatSPP:
            ch_in = ch_out = args[0]
            args = [ch_in, ch_out, *args[1:]]
            for idx, x in enumerate(deepcopy(f)):
                if x == "Backbone":
                    f.pop(idx)
                    for c_fr_idx, t in enumerate(connect_fr):
                        # if t != len(ch):
                        if t != si - 1:
                            f.append(t)
                    ch_res_rule = 0
                elif x == "PreLayer":
                    f.pop(idx)
                    # for c_fr_idx, t in enumerate([3, 7]):
                    for c_fr_idx, t in enumerate(nd['fpn-to-pan']):
                        if t != i - 1:
                            # prevent previous layer from duplicating
                            f.append(t+si)
                    ch_res_rule = 1

            # refer_ch = [ch[index] for index in f if index != -1]
            refer_ch = []
            sz_in = sz[f[0]]
            for index in f:
                # if index != -1 and ch[index] not in refer_ch:
                if ch[index] not in refer_ch:
                    refer_ch.append(ch[index])
                if sz_in != sz[index]:
                    print('Concat must have all the same size inputs')
                    print(f'{sz_in} is not the same as {sz[index]}')

        m_ = m(*args)  # nn module
        t = str(m)[8:-2].replace('__main__.', '')
        split_t = t.split(".")
        t = split_t[-1]

        # (tenace comment) more flexible definition for neck channel
        # nd['neck-channel'] if they are explictly described in neck yaml file
        # bm.connect[x][1] (x=2,1,0) otherwise
        if nd.get('neck-channel'):
            neck_channel = nd['neck-channel']
        if hasattr(m_, 'arch_param_define'):
            if path_freezing:
                m_.arch_param_define(
                    prev_sz=sz[f[0]],
                    prev_ch=ch[f[0]],
                    f_sz=[sz[f_idx] for f_idx in f],
                    f_ch=[ch[f_idx] for f_idx in f],
                    # device=device,
                    refer_sz=refer_sz,
                    refer_ch=refer_ch,
                    neck_channel=neck_channel,  # nd['neck-channel'],
                    ch_res_rule=ch_res_rule,
                    fidx=f,
                    path_freezing=path_freezing
                )
            else:
                m_.arch_param_define(
                    prev_sz=sz[f[0]],
                    prev_ch=ch[f[0]],
                    f_sz=[sz[f_idx] for f_idx in f],
                    f_ch=[ch[f_idx] for f_idx in f],
                    # device=device,
                    refer_sz=refer_sz,
                    refer_ch=refer_ch,
                    neck_channel=neck_channel,  # nd['neck-channel'],
                    ch_res_rule=ch_res_rule,
                    fidx=f
                )
                # return_list.append(m_.arch_weight)
        params = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.params = si+i, f, t, params

        # LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s'
        #             % (i, f, n, params, t, args))
        # print(f'{m_.type} added: {len(list(m_.modules()))}')
        layers.append(m_)
        # ch[x] : input channel for layer-#x
        ch.append(ch_out)
        sz.append(sz_out)

    # connection between neck to head
    output = nd['neck-to-head']
    for k in range(len(output)):
        output[k] += si

    return nn.ModuleList([*layers]), ch, output, return_list


def parse_backbone(backbone_dict, bridge_cnt):
    bridge_id, bridge_ch = [], []
    channels = [3]  # assuming 3-ch(RGB) data input
    for i, (f, n, m, args) in enumerate(backbone_dict['backbone']):
        # from, number, module, arguments
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            args[j] = eval(a) if isinstance(a, str) else a
        if m in [Conv, Focus, CBR, CBL, CBS, CBM, CBR6, DBR6]:
            # (case I) 3rd argument = stride
            ch_in, ch_out = channels[f], args[0]
            if i > 0 and args[2] == 2:  # args[2]=stride==2; downsize
                bridge_id.append(i-1)
                bridge_ch.append(ch_in)
        elif m is MB:
            # (case II) 4th argument = stride
            ch_in, ch_out = channels[f], args[0]
            if i > 0 and args[3] == 2:  # args[3]=stride==2; downsize
                bridge_id.append(i-1)
                bridge_ch.append(ch_in)
        elif m in [CSP, SPP, Bottleneck]:
            # (case III) no stride
            ch_in, ch_out = channels[f], args[0]
        elif m is Concat:
            # (case IV) no stride, output ch = sum of input channels
            ch_out = sum([channels[x] for x in f])
        else:
            # (case V) no output channel
            ch_out = ch_in = channels[f]

        if i == len(backbone_dict['backbone'])-1:
            bridge_id.append(i)
            bridge_ch.append(ch_out)

        if i == 0:
            channels = []
        channels.append(ch_out)

    return bridge_id[-bridge_cnt:], bridge_ch[-bridge_cnt:]


def parse_neck(br_id, br_ch, br_cnt, backbone_len, neck_dict, nc, anchors):
    # neck
    neck_arch = neck_dict['neck']
    for i, (f, n, m, args) in enumerate(neck_arch):
        if isinstance(f, list):
            for j, fr in enumerate(f):
                if isinstance(fr, str):
                    if fr == 'Backbone':
                        fr = br_id.pop(-2)
                    elif fr == 'PreLayer':
                        pass
                    else:
                        print(fr)
                        raise NotImplemented
                    f[j] = fr
                else:
                    f[j] = fr + backbone_len if fr != -1 else fr
        else:
            f = f + backbone_len if f != -1 else f
        neck_arch[i] = [f, n, m, args]

    # head
    neck_exit = neck_dict.pop('neck-to-head')
    for i, ex in enumerate(neck_exit):
        neck_exit[i] += backbone_len
    head = [neck_exit, 1, 'Detect', [nc, anchors]]
    neck_dict['neck'].append(head)


def parse_super_neck(connect_fr, ch, br_cnt, si,
                     neck_dict, nc, anchors):
    """
        si: starting index of 'neck'
        connect_fr: index of backbone layers connecting neck
        ch: output channel of backbone layers connecting neck
    """
    neck_arch = neck_dict['neck']
    for i, (f, n, m, args) in enumerate(neck_arch):
        """ from, number, module, arguments """
        m = eval(m) if isinstance(m, str) else m  # nn module name
        for j, a in enumerate(args):
            try:
                # arguments (list)
                args[j] = eval(a) if isinstance(a, str) else a
            except BaseException:
                pass

        if m in [Concat, ConcatForNas]:
            for idx, x in enumerate(f):
                if x == "Backbone":
                    f.pop(idx)
                    for fr in connect_fr:
                        f.append(fr)
                elif x == "PreLayer":
                    f.pop(idx)
                    for c_fr_idx, t in enumerate([3, 7]):
                        f.append(t+si)
                    ch_res_rule = 1
            refer_ch = [ch[index] for index in f if index != -1]

        neck_arch[i] = [f, n, m, args]

    # head
    neck_exit = neck_dict.pop('neck-to-head')
    for i, ex in enumerate(neck_exit):
        neck_exit[i] += si
    head = [neck_exit, 1, 'Detect', [nc, anchors]]
    neck_dict['neck'].append(head)


def build_model(d):
    LOGGER.info('%3s%18s%3s%10s  %-40s%-30s'
                % ('', 'from', 'n', 'params', 'module', 'arguments'))
    ch = [3]

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['neck']):
        # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except BaseException:
                pass

        # n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, Focus, CSP,  # C3,
                 CBR, CBS, CBL, CBM, CBR6, DBR6]:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
            if m in [CSP]:  # C3
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is MB:
            args.insert(0, ch[f])
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])  # ch_in
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        # build module
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) \
            if n > 1 else m(*args)

        # atttr: module type
        t = str(m)[8:-2].replace('__main__.', '')
        # split_t = t.split(".")
        # t = split_t[-1]
        # attr: number params
        np = sum([x.numel() for x in m_.parameters()])
        # attach attributes: index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np

        # print
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s'
                    % (i, f, n, np, t, args))

        # append to savelist
        save.extend(x % i for x
                    in ([f] if isinstance(f, int) else f) if x != -1)
        # print(f'{m_.type} added: {len(list(m_.modules()))}')
        layers.append(m_)

        # append output channel
        if i == 0:  # remove input channel
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
