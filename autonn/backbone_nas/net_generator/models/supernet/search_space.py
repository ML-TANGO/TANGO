'''
Once for All: Train One Network and Specialize it for Efficient Deployment
Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
International Conference on Learning Representations (ICLR), 2020.
It is modified by Wonseon-Lim and based on Once for ALL Paper code.
'''

import copy
import random
from collections import OrderedDict

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import \
    DynamicMBConvLayer
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import MyNetwork, make_divisible, val2list
from ofa.utils.layers import (ConvLayer, IdentityLayer, LinearLayer,
                              MBConvLayer, ResidualBlock)

__all__ = ["BackBoneMobileNetV3"]


class BackBoneMobileNetV3(MobileNetV3):
    '''
    define backbone
    '''
    def __init__(self,
                 bn_param=(0.1, 1e-5),
                 dropout_rate=0.1,
                 args=None
                 ):
        if args is None:
            args.width_mult = 1.0
            args.ks_list = 3
            args.expand_ratio_list = 6
            args.depth_list = 4
            args.n_classes = 1000

        self.return_layers = None

        args.ks_list = val2list(args.ks_list, 1)
        args.expand_ratio_list = val2list(args.expand_ratio_list, 1)
        args.depth_list = val2list(args.depth_list, 1)

        args.ks_list.sort()
        args.expand_ratio_list.sort()
        args.depth_list.sort()
        args.bn_param = bn_param
        args.dropout_rate = dropout_rate
        args.base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

        self.args = args

        args.final_expand_width = make_divisible(
            args.base_stage_width[-2] * args.width_mult,
            MyNetwork.CHANNEL_DIVISIBLE
        )
        args.last_channel = make_divisible(
            args.base_stage_width[-1] * args.width_mult,
            MyNetwork.CHANNEL_DIVISIBLE
        )

        args.stride_stages = [1, 2, 2, 2, 1, 2]
        args.act_stages = ["relu", "relu", "relu",
                           "h_swish", "h_swish", "h_swish"]
        args.se_stages = [False, False, True, False, True, True]
        # [2,3,4] => [4, 4, 4, 4, 4]
        args.n_block_list = [1] + [max(args.depth_list)] * 5
        args.width_list = []
        for base_width in args.base_stage_width[:-2]:
            width = make_divisible(
                base_width * args.width_mult, MyNetwork.CHANNEL_DIVISIBLE
            )
            args.width_list.append(width)

        args.input_channel, args.first_block_dim = \
            args.width_list[0], args.width_list[1]

        (first_conv, blocks, final_expand_layer,
         feature_mix_layer, classifier) = self.make_layers(args)

        super(BackBoneMobileNetV3, self).__init__(
            first_conv, blocks, final_expand_layer,
            feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(momentum=self.args.bn_param[0],
                          eps=self.args.bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx)
                              for block_idx in self.block_group_info]

    def make_layers(self, args):
        '''
        make layers
        '''
        # first conv layer
        first_conv = ConvLayer(
            3, args.input_channel, kernel_size=3, stride=2, act_func="h_swish"
        )
        first_block_conv = MBConvLayer(
            in_channels=args.input_channel,
            out_channels=args.first_block_dim,
            kernel_size=3,
            stride=args.stride_stages[0],
            expand_ratio=1,
            act_func=args.act_stages[0],
            use_se=args.se_stages[0],
        )
        first_block = ResidualBlock(
            first_block_conv,
            IdentityLayer(args.first_block_dim, args.first_block_dim)
            if args.input_channel == args.first_block_dim
            else None,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        args.block_index = 1
        args.feature_dim = args.first_block_dim

        for args.width, args.n_block, args.s, args.act_func, args.use_se \
            in zip(args.width_list[2:],
                   args.n_block_list[1:],  # [4, 4, 4, 4, 4]
                   args.stride_stages[1:],  # [2, 2, 2, 1, 2]
                   args.act_stages[1:],
                   args.se_stages[1:]):
            self.block_group_info.append(
                [args.block_index + i for i in range(args.n_block)])
            args.block_index += args.n_block

            args.output_channel = args.width
            for i in range(args.n_block):
                if i == 0:
                    args.stride = args.s
                else:
                    args.stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(args.feature_dim),
                    out_channel_list=val2list(args.output_channel),
                    kernel_size_list=self.args.ks_list,
                    expand_ratio_list=self.args.expand_ratio_list,
                    stride=args.stride,
                    act_func=args.act_func,
                    use_se=args.use_se,
                )
                if ((args.stride == 1 and
                     args.feature_dim == args.output_channel)):
                    args.shortcut = IdentityLayer(args.feature_dim,
                                                  args.feature_dim)
                else:
                    args.shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv,
                                            args.shortcut))
                args.feature_dim = args.output_channel
        # final expand layer, feature mix layer & classifier
        final_expand_layer = ConvLayer(
            args.feature_dim, args.final_expand_width,
            kernel_size=1, act_func="h_swish"
        )
        feature_mix_layer = ConvLayer(
            args.final_expand_width,
            args.last_channel,
            kernel_size=1,
            bias=False,
            use_bn=False,
            act_func="h_swish",
        )

        classifier = LinearLayer(
            args.last_channel, self.args.n_classes,
            dropout_rate=self.args.dropout_rate)

        return (first_conv, blocks, final_expand_layer,
                feature_mix_layer, classifier)

    @staticmethod
    def name():
        '''
        get net name
        '''
        return "BackBoneMobileNetV3"

    def forward(self, x):
        '''
        forward
        '''
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]  # depth ex) [2, 3, 2, 3, 2]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(
            2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

    def forward_fpn(self, _x):
        '''
        compute fpn
        '''
        # FPN outputs
        outputs = OrderedDict()

        # first conv
        _x = self.first_conv(_x)
        # first block
        _x = self.blocks[0](_x)
        # blocks
        # block_idx ex) [1,2,3,4,5]
        for stage_id, block_idx in enumerate(self.block_group_info):
            # runtime_depth ex) [2, 3, 2, 3, 2]
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for _, idx in enumerate(active_idx):
                _x = self.blocks[idx](_x)
                # only use first layer's feature in each stage
                if ((self.return_layers is not None and
                     idx in self.return_layers)):
                    outputs[str(idx)] = _x
                elif (self.return_layers is None
                      and idx == self.block_group_info[-1][-1]):
                    outputs[str(idx)] = _x
        return outputs, list(outputs.keys())

    @property
    def module_str(self):
        '''
        get representation for module
        '''
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"

        _str += self.final_expand_layer.module_str + "\n"
        _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        '''
        get config
        '''
        return {
            "name": BackBoneMobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "final_expand_layer": self.final_expand_layer.config,
            "feature_mix_layer": self.feature_mix_layer.config,
            "classifier": self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        '''
        bulid net
        '''
        raise ValueError("do not support this function")

    @property
    def grouped_block_index(self):
        '''
        get block index
        '''
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        '''
        convert state
        '''
        model_dict = state_dict()
        for key in state_dict:
            if ".mobile_inverted_conv." in key:
                new_key = key.replace(".mobile_inverted_conv.", ".conv.")
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif ".bn.bn." in new_key:
                new_key = new_key.replace(".bn.bn.", ".bn.")
            elif ".conv.conv.weight" in new_key:
                new_key = new_key.replace(".conv.conv.weight", ".conv.weight")
            elif ".linear.linear." in new_key:
                new_key = new_key.replace(".linear.linear.", ".linear.")
            ########################################
            elif ".linear." in new_key:
                new_key = new_key.replace(".linear.", ".linear.linear.")
            elif "bn." in new_key:
                new_key = new_key.replace("bn.", "bn.bn.")
            elif "conv.weight" in new_key:
                new_key = new_key.replace("conv.weight", "conv.conv.weight")
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, "%s" % new_key
            model_dict[new_key] = state_dict[key]
        super(BackBoneMobileNetV3, self).load_state_dict(model_dict, **kwargs)

    def set_max_net(self):
        '''
        set max net
        '''
        self.set_active_subnet(
            _ks=max(self.ks_list),
            _e=max(self.expand_ratio_list),
            _d=max(self.depth_list)
        )

    def set_active_subnet(self, _ks=None, _e=None, _d=None):
        '''
        set active subnet
        '''
        _ks = val2list(_ks, len(self.blocks) - 1)
        expand_ratio = val2list(_e, len(self.blocks) - 1)
        depth = val2list(_d, len(self.block_group_info))

        for block, _k, _e in zip(self.blocks[1:], _ks, expand_ratio):
            if _k is not None:
                block.conv.active_kernel_size = _k
            if _e is not None:
                block.conv.active_expand_ratio = _e

        for i, _d in enumerate(depth):
            if _d is not None:
                self.runtime_depth[i] = min(
                    len(self.block_group_info[i]), _d)  # min(5, d)

    def set_constraint(self, include_list, constraint_type="depth"):
        '''
        set constraint
        '''
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        '''
        clear constraint
        '''
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None

    def sample_active_subnet(self):
        '''
        sampling subnet
        '''
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [
                ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [
                expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            _e = random.choice(e_set)
            expand_setting.append(_e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            _d = random.choice(d_set)
            depth_setting.append(_d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        '''
        get subnet
        '''
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        final_expand_layer = copy.deepcopy(self.final_expand_layer)
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    ResidualBlock(
                        self.blocks[idx].conv.get_active_subnet(
                            input_channel, preserve_weight
                        ),
                        copy.deepcopy(self.blocks[idx].shortcut),
                    )
                )
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(
            first_conv, blocks, final_expand_layer,
            feature_mix_layer, classifier
        )
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        '''
        get net from config
        '''
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config["conv"]["out_channels"]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    {
                        "name": ResidualBlock.__name__,
                        "conv": self.blocks[idx].conv.get_active_subnet_config(
                            input_channel
                        ),
                        "shortcut": (self.blocks[idx].shortcut.config
                                     if self.blocks[idx].shortcut is not None
                                     else None),
                    }
                )
                input_channel = self.blocks[idx].conv.active_out_channel
            block_config_list += stage_blocks

        return {
            "name": MobileNetV3.__name__,
            "bn": self.get_bn_param(),
            "first_conv": first_conv_config,
            "blocks": block_config_list,
            "final_expand_layer": final_expand_config,
            "feature_mix_layer": feature_mix_layer_config,
            "classifier": classifier_config,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        '''
        modify hidden weights
        '''
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
