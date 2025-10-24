import os
import yaml
from PIL import Image
import pandas as pd
import numpy as np
import requests
import albumentations as A
from torch.utils.data.sampler import SubsetRandomSampler
import classification_settings


def yml_to_dict(filepath):
    with open(filepath) as f:
        taskdict = yaml.load(f, Loader=yaml.FullLoader)
    return taskdict


# load image on dataloader with grayscale
def custom_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        return img.convert("L")


def train_val_split(dataset):
    # test set 을 데이터셋에서 나누려고 임시로 수정함
    dataset_size = len(dataset)  # 전체크기
    indices = list(range(dataset_size))  # 전체 인덱스 리스트만들고
    split = int(np.floor(0.2 * dataset_size))  # 내림함수로 20% 지점 인덱스
    np.random.seed(22)
    np.random.shuffle(indices)  # 인덱스 리스트 섞어줌

    # 섞어진 리스트에서 처음부터 ~번째 까지 val, ~+1번째부터 끝 인덱스까지 train
    train_indices_, test_indices = indices[split:], indices[:split]

    # for snu dataset
    if classification_settings.train_root == "/snu_xray-resize/":
        np.random.seed(42)
        np.random.shuffle(train_indices_)
        split2 = int(np.floor(0.2 * len(train_indices_)))
        train_indices, val_indices = train_indices_[split2:], train_indices_[:split2]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
    

    train_sampler = SubsetRandomSampler(train_indices_)
    val_sampler = SubsetRandomSampler(test_indices)
    test_sampler = None
    result_dict = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

    return result_dict


def align_csv(path, align_list):
    # path = "./result.csv"
    data_frame = pd.read_csv(path)
    data_frame = data_frame[align_list]
    data_frame.to_csv(path, mode="w")
    return data_frame


def save_csv(data, path):
    df = pd.DataFrame(data)
    df = df.transpose()
    if not os.path.exists(path):
        df.to_csv(path, mode="w")
    else:
        df.to_csv(path, mode="a", header=False)


def send_alarm_to_slack(msg):
    url = ""

    payload = {"text": msg}

    requests.post(url, json=payload)


def create_directory(_dir="./runs/"):
    num = 1
    while True:
        temp = _dir + "exp" + str(num)
        if not os.path.exists(temp):
            os.makedirs(temp)
            break
        else:
            num += 1

    return temp


# apply albumentations on torch dataloader
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


if __name__ == "__main__":
    send_alarm_to_slack("test")
