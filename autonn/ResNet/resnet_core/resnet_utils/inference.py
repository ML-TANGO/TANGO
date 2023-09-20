#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Medical Chest X-ray classification Test
# 2023 08 28 by Seongwon Jo
###########################################################################
_description = '''\
====================================================
inference.py : T
                    Written by Seongwon Jo @ 2023-08-28
====================================================
Example : python inference.py -i './test/' -o 'all' -p 'weights.pt' -m resnet18
          python inference.py -i './test/NORMAL/1.jpeg' -o 'one' -p 'weights.pt' -m resnet18
          python inference.py -i './test/NORMAL/' -o 'onebyone' -p 'weights.pt' -m resnet18 --save_dict 
'''
import argparse
import textwrap
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pprint
import pandas as pd
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.utils import yml_to_dict, custom_pil_loader, Transforms, save_csv
from setup import Initializer, MyDataset
from trainer import evaluate
import classification_settings


def preprocess_image(
    img: np.ndarray, mean=[
        0.5,], std=[
            0.5,]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def ArgumentParse(_intro_msg, L_Param, bUseParam=False):
    parser = argparse.ArgumentParser(
        prog='inference.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    # parser.add_argument('-y', '--yml_path', default="./train_options.yml",
    #                     help="path to yml file contains options")
    parser.add_argument('-i', '--image_path', default="./test2")
    parser.add_argument('-o', '--order', default="all")
    parser.add_argument('-d', '--device', default="cuda:0")
    parser.add_argument('-p', '--pt_path', required=True ,help="please input pt file path")
    parser.add_argument('-m', '--model', required=True ,help="what model ?")
    parser.add_argument('--save_dict', action='store_true')

    args = parser.parse_args()
    return args


class Inference:
    def __init__(self, L_Param=None):
        self.args       = ArgumentParse(_description, L_Param, bUseParam=False)
        # self.args       = ArgumentParse()
        # self.o_dict     = yml_to_dict(self.args.yml_path)['parameters']

        _model, _device = self.initialization(device=self.args.device)
        self.model      = _model
        self.device     = _device
        print(f"\nOperate selected inference ... '{self.args.order}'\n")

    def initialization(self, device):
        initializer = Initializer(net=self.args.model,
                                  dataset="custom",
                                  device_num=device)
        model = initializer.model
        device = initializer.device
        model.load_state_dict(torch.load(self.args.pt_path,
                                         map_location=device)['model_state_dict'], strict=True)

        return model, device

    @staticmethod
    def loader_with_transforms(root="./test", batch_size=16):
        my_dataset_root = root
        # _transforms = classification_settings.transforms_T
        _transforms = Transforms(classification_settings.custom_test)

        # data_path == image folder path
        dataset = datasets.ImageFolder(root=my_dataset_root,
                                       transform=_transforms, loader=custom_pil_loader)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)

        return data_loader

    # class 폴더에 분류가 되어있는 경우
    def operation_all(self):
        data_loader = self.loader_with_transforms(root=self.args.image_path)
        test_accuracy, report , _l = evaluate(self.model, data_loader, self.device, is_test=True)
        print("\n=============================================")
        # pprint.pprint(report)
        print('prediction Accuracy: {:.2f}%'.format(report['accuracy']*100))
        print("=============================================\n")
        print('details')
        print("---------------------------------------------")
        print('NORMAL')
        pprint.pprint(report['NORMAL'])
        print("\n")
        print('PNEUMONIA')
        pprint.pprint(report['PNEUMONIA'])
        print("=============================================")


    def operation_onebyone(self):
        _transforms = Transforms(classification_settings.custom_test)

        f_list = os.listdir(self.args.image_path)
        transformed_list = []
        for file in f_list:
            img = custom_pil_loader(self.args.image_path + '/' + file)
            transformed_list.append(_transforms(img))

        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        r_list = []

        self.model.eval()

        with torch.no_grad():
            for data in transformed_list:
                output = self.model(data.unsqueeze(0).float().to(self.device))

                _score, predicted = output.max(1)
                pred = label_tags[predicted.item()]

                # _score_p = torch.softmax(output).max(1)
                r_list.append([pred, torch.softmax(output, dim=1).max().cpu()])

        r_dict = dict(zip(f_list, r_list))
        path = './infer_result.csv'
        if self.args.save_dict:
            df = pd.DataFrame(r_dict)
            df = df.transpose()
            df.columns = ['predict', 'softmax value']
            if not os.path.exists(path):
                df.to_csv(path, mode="w")
            else:
                df.to_csv(path, mode="a", header=False)

        # print('prediction result: {file : [prediction, score]}\n')
        # print("="*80)
        # pprint.pprint(r_dict)
        # print("="*80)

        n = 0
        p = 0
        for k, v in r_dict.items():
            if v[0] == "Normal":
                n += 1
            else:
                p += 1

        print("# of images: ", len(r_dict), "\nNormal: ", n, "\nPneumonia: ", p)

    def operation_one_image(self):
        _transforms = Transforms(classification_settings.custom_test)
        img = custom_pil_loader(self.args.image_path)
        transformed = _transforms(img)

        label_tags = {
            0: 'Normal',
            1: 'Pneumonia',
        }
        self.model.eval()

        with torch.no_grad():
            output = self.model(transformed.unsqueeze(0).float().to(self.device))

            _score, predicted = output.max(1)
            pred = label_tags[predicted.item()]

        print('prediction result: %s  Score: %f' %(pred, _score))
        print('softmax result : ', torch.softmax(output, dim=1).max())

    def selected_operation(self,):
        if self.args.order == 'all':
            self.operation_all()
        elif self.args.order == "onebyone":
            self.operation_onebyone()
        elif self.args.order == "one":
            self.operation_one_image()
        else:
            print("Please check 'operation' input.")
        # ops = {
        #     'all': self.operation_all(),
        #     'onebyone': self.operation_onebyone(),
        #     'one': self.operation_one_image()
        # }.get(self.args.order, "Please check 'operation' input.")




if __name__ == '__main__':
    infer = Inference()
    infer.selected_operation()

