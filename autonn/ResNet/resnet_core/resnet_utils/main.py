"""autonn/ResNet/resnet_core/resnet_utils/main.py
This code not used in the project.
"""
import argparse

from utils.utils import send_alarm_to_slack
from utils.yml_to_tasklist import yml_to_tasklist
from trainer import Training


def main():
    args = argument()
    select_task(args)


def argument():
    parser = argparse.ArgumentParser(description="PyTorch Image Classification")
    parser.add_argument(
        "-y",
        "--yml_path",
        default="./train_options.yml",
        help="path to yml file contains experiment options",
    )
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-c", "--csv_name", default="result.csv")
    parser.add_argument("-t", "--task", default="train")
    parser.add_argument("-a", "--alarm", default=False)
    # parser.add_argument('-n', "--case_name", default='test1')
    arguments = parser.parse_args()
    return arguments


def sequential_task(tasklist, device, csv_name, alarm):
    for i in tasklist:
        trainer = Training(
            tasklist[i], index_num=i, device_num=device, csv_path=csv_name, alarm=alarm
        )
        trainer.operation()
        trainer.finish()
    if alarm:
        send_alarm_to_slack("All task done")


def inference_task(yml_path):
    print("will be implemented")


def select_task(arguments):
    if arguments.task == "train":
        tasklist = yml_to_tasklist(arguments.yml_path)
        sequential_task(tasklist, arguments.device, arguments.csv_name, arguments.alarm)
    elif arguments.task == "inference":
        inference_task(arguments.yml_path)


if __name__ == "__main__":
    main()
