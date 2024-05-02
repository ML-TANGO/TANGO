"""
high level support for doing this and that.
"""


class CDefaults:
    '''
    default class
    '''

    def __init__(self):
        self.combine_deflist = []
        self.conv_deflist = []
        self.norm_deflist = []
        self.pooling_deflist = []
        self.padding_deflist = []
        self.activation_deflist = []
        self.spp_deflist = []
        self.loss_deflist = []
        self.head_deflist = []
        self.etc_deflist = []

        self.tags = {
            'Conv2d': {'in_channels': 1, 'out_channels': 1,
                       'kernel_size': (3, 3),
                       'stride': (1, 1), 'padding': (0, 0), 'bias': True},
            'MaxPool2d': {'kernel_size': (2, 2), 'stride': (2, 2),
                          'padding': (0, 0),
                          'dilation': 1, 'return_indices': False,
                          'ceil_mode': False},
            'AvgPool2d': {'kernel_size': (2, 2),
                          'stride': (2, 2),
                          'padding': (0, 0)},
            'AdaptiveAvgPool2d': {'output_size': (1, 1)},
            'MP': {'k': 2},
            'SP': {'kernel_size': (3, 3), 'stride': (1, 1)},
            'ZeroPad2d': {'padding': 1},
            'ConstantPad2d': {'padding': 2, 'value': 3.5},

            'ReLU': {'inplace': False},  # 10-13 inplace
            'ReLU6': {'inplace': False},
            'Sigmoid': {},
            'LeakyReLU': {'negative_slope': 0.01, 'inplace': False},
            'Tanh': {},
            'SoftMax': {'dim': 0},

            'BatchNorm2d': {'num_features': 1},

            'Linear': {'in_features': 1, 'out_features': 1, 'bias': True,
                       'device': None, 'dtype': None},

            'Dropout': {'p': 0.5, 'inplace': False},

            'BCELoss': {'weight': None, 'size_average': True,
                        'reduce': True, 'reduction': 'mean'},
            'CrossEntropyLoss': {'weight': None, 'size_average': True,
                                 'ignore_index': None,
                                 'reduce': True, 'reduction': 'mean',
                                 'label_smoothing': 0.0},
            'MSELoss': {'size_average': True,
                        'reduce': True, 'reduction': 'mean'},

            'Flatten': {'start_dim': 1, 'end_dim': - 1},

            'Upsample': {'size': None, 'scale_factor': None,
                         'mode': 'nearest',
                         'align_corners': None,
                         'recompute_scale_factor': None},

            'Bottleneck': {'inplanes': 1, 'planes': 1,
                           'stride': (1, 1), 'downsample':False,
                           'groups': 1, 'base_width': 64,
                           'dilation': 1, 'norm_layer': None},

            'BasicBlock': {'inplanes': 1, 'planes': 1,
                           'stride': (1, 1), 'downsample':False,
                           'groups': 1, 'base_width': 64,
                           'dilation': 1, 'norm_layer': None},
            'Concat': {'dim': 1},
            'DownC': {'in_channels': 64, 'out_channels': 64,
                      'n': 1, 'kernel_size': (2, 2)},
            'SPPCSPC': {'in_channels': 64, 'out_channels': 64,
                        'n': 1, 'shortcut': False, 'groups': 1,
                        'expansion': 0.5, 'kernels': (5, 9, 13)},
            'ReOrg': {},
            'Conv': {'in_channels': 64, 'out_channels': 64,
                     'kernel_size': 1, 'stride': 1, 'padding': None, 'groups': 1,
                     'act': True},
            'IDetect': {'nc': 80, 'anchors': (), 'ch': ()}
        }

        self.conv_deflist = ['Conv2d', 'Conv']
        self.pooling_deflist = ['MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d'
                                'MP', 'SP']
        self.padding_deflist = ['ZeroPad2d', 'ConstantPad2d']
        self.activation_deflist = ['ReLU', 'ReLU6', 'Sigmoid',
                                   'LeakyReLU', 'Tanh', 'Softmax']
        self.norm_deflist = ['BatchNorm2d']
        self.loss_deflist = ['BCELoss', 'CrossEntropyLoss', 'MSELoss']
        self.etc_deflist = ['Sequential', 'Flatten', 'Upsample',
                            'Dropout', 'Linear',
                            'Bottleneck', 'BasicBlock', 'ReOrg']
        self.combine_deflist = ['Concat']
        self.spp_deflist = ['DonwC', 'SPPCSPC']
        self.head_deflist = ['IDetect']

    def pooling_layer(self):
        '''
        print pooling layers
        '''
        print(f"{self.pooling_deflist}")

    def loss_funtion(self):
        '''
        print loss functions
        '''
        print(f"{self.loss_deflist}")
        print(f"{self.loss_deflist}")
