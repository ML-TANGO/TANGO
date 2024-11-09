import torch
from torchvision import transforms
from PIL import Image

from .resnet152 import ResNet152 as ResNet152Model
from .densenet201 import DenseNet201 as DenseNet201Model

from typing import Tuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet152(object):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )

    CLASS_NAMES = ["Normal", "Pneumonia"]

    def __init__(
        self,
        checkpoint_path: str = "pretained_model/kagglecxr_resnet152_normalize.pt",
    ):
        self.__model = ResNet152Model(num_classes=2)
        load = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )["model_state_dict"]
        self.__model.load_state_dict(
            load,
            strict=True,
        )
        self.__model.to(DEVICE)
        self.__model.eval()

    def predict_image(self, image_path: str) -> Tuple[str, float]:
        img = Image.open(image_path).convert("L")
        img.load()
        img = self.preprocess(img)
        cls_label = "None"
        cls_prob = 0.0
        with torch.no_grad():
            output = self.__model(img.unsqueeze(0).float().to(DEVICE))
            result = output.max(1)
            cls_label = self.CLASS_NAMES[result.indices.item()]
            cls_prob = result.values.item()
        return cls_label, cls_prob

    def __call__(self, image_path: str) -> Tuple[str, float]:
        return self.predict_image(image_path)


class DenseNet201(object):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,), inplace=True),
        ]
    )

    CLASS_NAMES = ["Normal", "Pneumonia"]

    def __init__(
        self,
        checkpoint_path: str = "pretained_model/ckpt_densenet201.pt",
    ):
        self.__model = DenseNet201Model(num_classes=2)
        load = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )["model_state_dict"]
        self.__model.load_state_dict(
            load,
            strict=True,
        )
        self.__model.to(DEVICE)
        self.__model.eval()

    def predict_image(self, image_path: str) -> Tuple[str, float]:
        img = Image.open(image_path).convert("L")
        img.load()
        img = self.preprocess(img)
        cls_label = "None"
        cls_prob = 0.0
        with torch.no_grad():
            output = self.__model(img.unsqueeze(0).float().to(DEVICE))
            result = output.max(1)
            cls_label = self.CLASS_NAMES[result.indices.item()]
            cls_prob = result.values.item()
        return cls_label, cls_prob

    def __call__(self, image_path: str) -> Tuple[str, float]:
        return self.predict_image(image_path)
