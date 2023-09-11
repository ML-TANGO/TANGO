#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Medical Chest X-ray classification Test
# Working Directory : D:\Work_2022\work_space\
# 2020 05 31 by Seongwon Jo
###########################################################################
_description = '''\
====================================================
inference_folder.py : T
                    Written by Seongwon Jo @ 2022-05-31
====================================================
Example : python inference_folder.py -i ./test 
'''
import os
import yaml
import argparse
import textwrap
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet
from models import densenet_1ch


def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='inference_folder.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-y', '--yml_path', default="./inference_settings.yml",
                        help="path to yml file contains options")
    parser.add_argument('-i', '--image_path', default="./test")

    args = parser.parse_args()
    return args


class image_classification:
    def __init__(self, L_Param=None):
        self.args       = ArgumentParse(_description, L_Param, bUseParam=False)

        self.o_dict     = self.yml_to_dict(self.args.yml_path)
        _model, _device = self.initialization(self.o_dict)
        self.model      = _model
        self.device     = _device

    def yml_to_dict(self, filepath):
        with open(filepath) as f:
            taskdict = yaml.load(f, Loader=yaml.FullLoader)
        return taskdict

    # load image on dataloader with grayscale
    def custom_pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()
            return img.convert('L')

    # resnet200
    def resnet200(self, pretrained: bool = False, progress: bool = True, **kwargs) -> resnet.ResNet:
        return resnet._resnet('resnet200', resnet.Bottleneck, [3, 24, 36, 3], pretrained, progress,
                       **kwargs)

    # select model
    def select_model(self, net, dataset, device_num):
        num_classes = {
            'custom': 2,
        }.get(dataset, "error")

        model = {
            'resnet18': resnet.resnet18(num_classes=num_classes),
            'resnet50': resnet.resnet50(num_classes=num_classes),
            'resnet101': resnet.resnet101(num_classes=num_classes),
            'resnet152': resnet.resnet152(num_classes=num_classes),
            'resnet200': self.resnet200(num_classes=num_classes),
            'densenet121': densenet_1ch.densenet121(num_classes=num_classes),
            'densenet169': densenet_1ch.densenet169(num_classes=num_classes),
            'densenet201': densenet_1ch.densenet201(num_classes=num_classes),
        }.get(net, "error")

        if model == "error":
            raise ValueError('please check your "net" input.')

        if net[0:6] == 'resnet':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        device = torch.device(device_num if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("device:", device)

        return model, device

    def transformed(self,):
        transforms = T.Compose([
            T.Resize((self.o_dict['resolution'], self.o_dict['resolution'])),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

        f_list = os.listdir(self.args.image_path)
        transformed_list = []
        for file in f_list:
            img = self.custom_pil_loader(self.args.image_path + '/' + file)
            transformed_list.append(transforms(img))

        return transformed_list

    def initialization(self, option_dict):
        model, device = self.select_model(option_dict['net'], option_dict['dataset'], option_dict['device'])
        model.load_state_dict(torch.load(option_dict['pt_path'], map_location=device)['model_state_dict'], strict=False)

        return model, device

    def evaluate_onebyone(self, data_list):
        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        r_list = []

        self.model.eval()

        with torch.no_grad():
            for data in data_list:
                output = self.model(data.unsqueeze(0).to(self.device))

                _score, predicted = output.max(1)
                pred = label_tags[predicted.item()]
                r_list.append([pred, _score.item()])

        return r_list

    def result_dict(self, result_list):
        f_list = os.listdir(self.args.image_path)
        r_dict = dict(zip(f_list, result_list))
        return r_dict


if __name__ == '__main__':
    c_Imgc = image_classification()

    transformed_list = c_Imgc.transformed()
    result_list      = c_Imgc.evaluate_onebyone(transformed_list)
    result_dict      = c_Imgc.result_dict(result_list)

    print('prediction result: {file : [prediction, score]}\n', result_dict)