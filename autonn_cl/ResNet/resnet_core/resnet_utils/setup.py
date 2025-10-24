"""autonn/ResNet/resnet_core/resnet_utils/setup.py
This code not used in the project.
"""
from email.policy import strict
import torch
import torch.nn as nn

from torch import optim
from torchvision import datasets
from torchvision.models import resnet, vgg, inception
from torch.utils.data import DataLoader
from torch.utils import model_zoo

from models import resnet_cifar10, densenet_1ch
from utils.utils import custom_pil_loader, Transforms, train_val_split
import classification_settings

import math
from torch.optim.lr_scheduler import _LRScheduler


# imagenet pretrained model
model_urls = {
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
}


class MyDataset:
    def __init__(
        self, *, data_src=classification_settings.data_folder, batch_size, dataset
    ):
        self.dataset = dataset
        self.data_src = data_src
        self.batch_size = batch_size

    # dataset select
    def load_dataset(self,):
        if self.dataset == "custom":
            my_dataset_root = {
                "train": self.data_src + classification_settings.train_root,
                "test": self.data_src + classification_settings.test_root,
            }
            my_transforms = {
                "train": Transforms(classification_settings.custom_transform),
                "test": Transforms(classification_settings.custom_test),
            }
            data_dict = {
                "train": datasets.ImageFolder(
                    root=my_dataset_root["train"],
                    transform=my_transforms["train"],
                    loader=custom_pil_loader,
                ),
                "val": datasets.ImageFolder(
                    root=my_dataset_root["train"],
                    transform=my_transforms["test"],
                    loader=custom_pil_loader,
                ),
                "test": datasets.ImageFolder(
                    root=my_dataset_root["test"],
                    transform=my_transforms["test"],
                    loader=custom_pil_loader,
                ),
            }

            print("dataset load: ", len(data_dict["train"]))
            sampler_dict = train_val_split(data_dict["train"])

            if sampler_dict["test"] != None:
                test_count = len(sampler_dict["test"])
            else:
                test_count = len(data_dict["test"]) 
                
            print(
                "train val split:\n",
                "train:",
                len(sampler_dict["train"]),
                "|",
                "val:",
                len(sampler_dict["val"]),
                "|",
                "test:",
                test_count
            )

            data_loaders = {
                x: torch.utils.data.DataLoader(
                    dataset=data_dict[x],
                    batch_size=self.batch_size,
                    sampler=sampler_dict[x],
                    pin_memory=True,
                    num_workers=4,
                )
                for x in ["train", "val", "test"]
            }
        else:
            raise ValueError('please check your "dataset" input.')

        return data_loaders["train"], data_loaders["val"], data_loaders["test"]


class Initializer:
    def __init__(self, net, lr=0.1, momentum=0.9, dataset=None, device_num="cuda:0"):
        self.net = net
        # self.lr = lr
        # self.momentum = momentum
        self.dataset = dataset
        self.device_num = device_num
        _model, _device = self.select_model(pretrained=classification_settings.use_pretrained)
        self.model = _model
        self.device = _device

        self.optimizer = MyOptimizer(self.model, lr, momentum)

    def select_optimizer(self, opt):
        switch_case = {
            "SGD": self.optimizer.SGD(),
            "Adam": self.optimizer.Adam(),
            "NAG": self.optimizer.Nesterov(),
            "RMSprop": self.optimizer.RMSprop(),
        }.get(opt, "error")

        if switch_case == "error":
            raise ValueError('please check your "opt" input.')
        return switch_case

    def select_model(self, pretrained=False):
        num_classes = {"custom": classification_settings.num_classes,}.get(
            self.dataset, "error"
        )

        model = {
            "resnet18": resnet.resnet18(num_classes=num_classes),
            "resnet50": resnet.resnet50(num_classes=num_classes),
            "resnet101": resnet.resnet101(num_classes=num_classes),
            "resnet152": resnet.resnet152(num_classes=num_classes),
            "vgg16": vgg.vgg16(num_classes=num_classes),
            "densenet121": densenet_1ch.densenet121(num_classes=num_classes),
            "densenet169": densenet_1ch.densenet169(num_classes=num_classes),
            "densenet201": densenet_1ch.densenet201(num_classes=num_classes),
            "densenet121_2048": densenet_1ch.densenet121_2048(num_classes=num_classes),
        }.get(self.net, "error")

        if model == "error":
            raise ValueError('please check your "net" input.')

        if self.dataset == "custom" and self.net[0:6] == "resnet":
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        if pretrained:
            self.model_ = model
            self._load_pretrained(url="TODO")
            model = self.model_
            print("Pretrained weight applied.")

        device = torch.device(self.device_num if torch.cuda.is_available() else "cpu")
        model.to(device)
        print("device:", device)

        # parallel processing (under construction)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model).cuda()

        return model, device

    def _load_pretrained(self, url, inchans=1):
        _ = url
        model_now = "mymodel"
        # state_dict = model_zoo.load_url(url)
        # state_dict = torch.hub.load_state_dict_from_url(url)
        ####
        # _pre_model = torch.load("./resnet50-0676ba61.pth")
        _pre_model = torch.load(classification_settings.pre_model_path, map_location='cuda:0')

        # for k,v in _pre_model.items():
        #     print(k)
        ####
        # _pre_model.fc = nn.Linear(2048, 2)
        pretrained_dict = _pre_model

        state_dict = pretrained_dict

        # # eliminate classifier weights
        # if model_now == "resnet":
        #     state_dict_ = {k: v for k, v in pretrained_dict.items() if k != "fc.weight"}
        #     state_dict = {k: v for k, v in state_dict_.items() if k != "fc.bias"}
        # elif model_now == "densenet":
        #     state_dict_ = {
        #         k: v for k, v in pretrained_dict.items() if k != "classifier.weight"
        #     }
        #     state_dict = {
        #         k: v for k, v in state_dict_.items() if k != "classifier.bias"
        #     }

        # if inchans == 1 and model_now == "resnet":
        #     conv1_weight = state_dict["conv1.weight"]
        #     state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
        # elif inchans == 1 and model_now == "densenet":
        #     conv0_weight = state_dict["features.conv0.weight"]
        #     state_dict["features.conv0.weight"] = conv0_weight.sum(dim=1, keepdim=True)
        # elif inchans != 1:
        #     assert False, "Invalid number of inchans for pretrained weights"
        self.model_.load_state_dict(state_dict, strict=False)

    def select_lossfunction(self, l_func):
        switch_case = {
            "CE": nn.CrossEntropyLoss(label_smoothing=0.1),
            "FL": FocalLoss(),
        }.get(l_func, "error")

        if switch_case == "error":
            raise ValueError("please check your (loss function) input.")
        return switch_case


class MyOptimizer:
    def __init__(self, model, lr, momentum):
        self.model = model
        self.lr = lr
        self.momentum = momentum

    def SGD(self):
        return optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0.0001,
        )

    def Adam(self):
        return optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0001
        )  # default lr = 0.001

    def Nesterov(self):
        return optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0.0005,
            nesterov=True,
        )

    def RMSprop(self):
        return optim.RMSprop(self.model.parameters(), lr=self.lr)  # default lr = 0.01


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss_ = nn.CrossEntropyLoss(reduction="none")
        ce_loss = ce_loss_(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def set_lr_scheduler(optimizer, epochs, last_ep):
    if last_ep == 0:
        last_ep = -1
    decay_step = int(epochs / 10)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    #                                               milestones=[60, 120, 160],
    #                                               gamma=0.2, last_epoch=last_ep)
    # lr_scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 1 / 2 ** (epoch // decay_step),
    #                        momentum_lambda=lambda epoch: 1 if (epoch // decay_step) == 0 else 9
    #                        if (epoch // decay_step) > 8 else 1 * (epoch // decay_step + 1))

    # lr_scheduler = optim.lr_scheduler.StepLR(
    #     optimizer=optimizer, step_size=decay_step, gamma=1, last_epoch=last_ep
    # )

    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=0)
    lr_scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=100, T_mult=2, eta_max=0.0002, T_up=10, gamma=0.5
    )
    return lr_scheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

