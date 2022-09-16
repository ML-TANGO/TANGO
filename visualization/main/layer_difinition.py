import sys
import os
import pickle
import getpass

import torch
import torch.nn as nn

class c_Defaults(object):
    def __init__(self):
        #self.defList = []

        self.conv_defList = []
        self.pooling_defList = []
        self.padding_defList = []
        self.activation_defList = []
        self.norm_defList = []
        self.linear_defList = []
        self.dropout_defList = []
        self.loss_defList = []
        self.utility_defList = []
        self.vision_defList = []
        self.etc_defList = []

        tags = {
            'Conv2d': {'in_channels': 1, 'out_channels': 1, 'kernel_size': (3, 3), 'stride': (1, 1), 'padding': (0, 0), 'bias': True},

            'MaxPool2d': {'kernel_size': (2, 2), 'stride': (2, 2), 'padding': (0, 0), 'dilation': 1, 'return_indices': False, 'ceil_mode': False},  # 10-13 dilation,return_indices,ceil_mode
            'AvgPool2d': {'kernel_size': (2, 2), 'stride': (2, 2), 'padding': (0, 0)},
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

            'Linear': {'in_features': 1, 'out_features': 1, 'bias': True, 'device': None, 'dtype': None},

            'Dropout': {'p': 0.5, 'inplace': False},

            'BCELoss': {'weight': None, 'size_average': True, 'reduce': True, 'reduction': 'mean'},
            'CrossEntropyLoss': {'weight': None, 'size_average': True, 'ignore_index': None, 'reduce': True, 'reduction': 'mean', 'label_smoothing': 0.0},
            'MSELoss': {'size_average': True, 'reduce': True, 'reduction': 'mean'},

            'Flatten': {'start_dim': 1, 'end_dim': - 1},

            'Upsample': {'size': None, 'scale_factor': None, 'mode': 'nearest', 'align_corners': None, 'recompute_scale_factor': None}
            }

        self.conv_defList = ['Conv2d']
        self.pooling_defList = ['MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d ']
        self.padding_defList = ['ZeroPad2d', 'ConstantPad2d']
        self.activation_defList = ['ReLU', 'ReLU6','Sigmoid', 'LeakyReLU','Tanh','Softmax']
        self.norm_defList = ['BatchNorm2d']
        self.linear_defList = ['Linear']
        self.dropout_defList = ['Dropout']
        self.loss_defList = ['BCELoss', 'CrossEntropyLoss', 'MSELoss']
        self.utility_defList = ['Flatten']
        self.vision_defList = ['Upsample']
        self.etc_defList = ['Sequential']

