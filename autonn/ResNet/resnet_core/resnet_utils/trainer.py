"""autonn/ResNet/resnet_core/resnet_utils/trainer.py
This code not used in the project.
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F

import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

from setup import set_lr_scheduler, MyDataset, Initializer
from utils.utils import align_csv, save_csv, send_alarm_to_slack, create_directory

import classification_settings

# test
def evaluate(model, test_loader, device, is_test=False, creterion=nn.CrossEntropyLoss(label_smoothing=0.1)):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    report = {}

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.to(device)
            output = model(data)

            loss = creterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            if is_test:
                labels_np = np.asarray(target.cpu())
                pred_np = np.asarray(predicted.cpu())

                for x in range(len(target)):
                    y_tr = labels_np[x]
                    y_true.append(y_tr)

                for x in range(len(data)):
                    y_pr = pred_np[x]
                    y_pred.append(y_pr)
    if is_test:
        if classification_settings.num_classes == 2:
            t_list = ["NORMAL", "PNEUMONIA"]
        else:
            t_list = ["BACTERIA", "NORMAL", "VIRUS"]
        report = classification_report(y_true, y_pred, target_names=t_list, output_dict=True)
        
    test_loss_ = test_loss / len(test_loader)    
    test_accuracy = 100.0 * correct / total

    return test_accuracy, report, test_loss_


class Training:
    def __init__(self, options_dict, index_num, device_num, csv_path, alarm):
        self.start_epoch = 1
        self.options_dict = options_dict
        self.index_num = index_num
        self.current_data = {}
        self.alarm = alarm

        initializer = Initializer(
            net=self.options_dict["net"],
            lr=self.options_dict["initial_lr"],
            momentum=0.9,
            dataset=self.options_dict["dataset"],
            device_num=device_num,
        )
        self.optimizer = initializer.select_optimizer(
            opt=self.options_dict["optimizer"]
        )
        self.model = initializer.model
        self.device = initializer.device

        self.creterion = initializer.select_lossfunction(l_func=self.options_dict["lossfunction"])

        dataset = MyDataset(
            data_src=classification_settings.data_folder,
            batch_size=self.options_dict["batch_size"],
            dataset=self.options_dict["dataset"],
        )
        self.train_loader, self.val_loader, self.test_loader = dataset.load_dataset()

        self.ckpt_info = "{}_{}_{}_{}_lr{}_{}".format(
            self.options_dict["dataset"],
            self.options_dict["net"],
            self.options_dict["optimizer"],
            self.options_dict["epochs"],
            self.options_dict["initial_lr"],
            # self.options_dict["initial_momentum"],
            datetime.today().strftime("%Y%m%d-%H%M"),
        )
        ckpt_dir = "./ckpt/" + self.ckpt_info
        self.writer = SummaryWriter(ckpt_dir)

        print("settings [{}]".format(self.ckpt_info))

        self._path = create_directory(_dir='./runs/')
        self.csv_path = self._path + '/' + csv_path  # './runs/exp1/result2.csv'


    # train
    def train(self,):
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.float().to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # loss = F.cross_entropy(output, target)
            loss = self.creterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = 100.0 * correct / total

        return train_loss, train_accuracy

    def operation(self,):
        # # load pt file
        # temp = '{}_{}_{}_{}_lr{}_m{}'.format(self.options_dict['dataset'], self.options_dict['net'],
        #                                     self.options_dict['optimizer'], 1000,
        #                                     self.options_dict['initial_lr'], self.options_dict['initial_momentum'] )
        # checkpoint = torch.load('/home/work/test1/ckpt/ckpt_'+temp+'.pt')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.start_epoch = checkpoint['epoch'] + 1
        # print("Loading checkpoint...\nstart ",checkpoint['epoch'],"epoch")

        # learning rate scheduler
        lr_scheduler = set_lr_scheduler(optimizer=self.optimizer,
                                        epochs=self.start_epoch + self.options_dict['epochs'],
                                        last_ep=self.start_epoch - 1)

        self.current_data[self.index_num] = {
            "batch_size": self.options_dict["batch_size"],
            "dataset": self.options_dict["dataset"],
            "model": self.options_dict["net"],
            "optimizer": self.options_dict["optimizer"],
            "initial_lr": self.options_dict["initial_lr"],
            # "initial_momentum": (
            #     self.options_dict["initial_momentum"]
            #     if self.options_dict["optimizer"] != "Adam"
            #     else ""
            # ),
        }

        print("training start")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.options_dict["epochs"] + 1):
            train_loss, train_accuracy = self.train()
            val_accuracy, _report, val_loss = evaluate(self.model, self.val_loader, self.device, creterion=self.creterion)
            
            if classification_settings.lr_scheduler:
                lr_scheduler.step()

            if epoch % 100 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self._path + "/ckpt_" + self.ckpt_info + "_{}epoch".format(epoch) + ".pt",
                )

                test_accuracy, report, test_loss = evaluate(
                    self.model, self.test_loader, self.device, is_test=True, creterion=self.creterion
                )
                save_csv(report, self._path + '/report.csv')

                self.current_data[self.index_num]["epochs"] = epoch
                self.current_data[self.index_num]["train acc"] = "{:.2f}".format(
                    train_accuracy
                )
                self.current_data[self.index_num]["val acc"] = "{:.2f}".format(
                    val_accuracy
                )
                self.current_data[self.index_num]["test acc"] = "{:.2f}".format(
                    test_accuracy
                )
                self.current_data[self.index_num]["test loss"] = "{:.2f}".format(
                    test_loss
                )
                self.current_data[self.index_num]["time"] = "{:.2f}".format(
                    (time.time() - start_time)
                )
                save_csv(self.current_data, self.csv_path)
                print(
                    "Train Accuracy: {:.2f}%, Val Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
                        train_accuracy, val_accuracy, test_accuracy
                    ),
                    epoch,
                    "epoch result saved",
                )
            else:
                print(
                    "[{}] Train loss: {:.4f}, Train Acc: {:.2f}%, Val Acc: {:.2f}%, time: {:.2f}s".format(
                        epoch,
                        train_loss,
                        train_accuracy,
                        val_accuracy,
                        (time.time() - start_time),
                    )
                )

            self.writer.add_scalar("train loss", train_loss, epoch)
            self.writer.add_scalar("val loss", val_loss, epoch)
            self.writer.add_scalar("train acc", train_accuracy, epoch)
            self.writer.add_scalar("val acc", val_accuracy, epoch)

    def finish(self):
        torch.cuda.empty_cache()
        align_csv(
            self.csv_path,
            [
                "model",
                "optimizer",
                "initial_lr",
                # "initial_momentum",
                "epochs",
                "train acc",
                "val acc",
                "test acc",
                "test loss",
                "time",
            ],
        )

        if self.alarm:
            send_alarm_to_slack(self.ckpt_info + " done")
        print("task done\n")
