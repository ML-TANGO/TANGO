"""
high level support for doing this and that.
"""


class CDefaults:
    '''
    default class
    '''

    def __init__(self):
        self.pooling_deflist = []
        self.padding_deflist = []
        self.activation_deflist = []
        self.norm_deflist = []
        self.loss_deflist = []
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
                         'recompute_scale_factor': None}
        }

        self.pooling_deflist = ['MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d']
        self.padding_deflist = ['ZeroPad2d', 'ConstantPad2d']
        self.activation_deflist = ['ReLU', 'ReLU6', 'Sigmoid',
                                   'LeakyReLU', 'Tanh', 'Softmax']
        self.norm_deflist = ['BatchNorm2d']
        self.loss_deflist = ['BCELoss', 'CrossEntropyLoss', 'MSELoss']
        self.etc_deflist = ['Sequential', 'Flatten', 'Upsample',
                            'Dropout', 'Linear', 'Conv2d']

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
