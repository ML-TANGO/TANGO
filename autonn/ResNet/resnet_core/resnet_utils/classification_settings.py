"""autonn/ResNet/resnet_core/resnet_utils/cam.py
This code not used in the project.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


num_classes = 2

data_folder = "/home/seongwon/PycharmProjects/data"
# "/chest_xray_balance/chest_xray/train" : balanced data | "/snu_xray-resize/" : snu_dataset | /chest_xray/ : kaggle
train_root = "/chest_xray/train/"
test_root = "/chest_xray/test/"

# lr scheduler
lr_scheduler = True

# load state dict from model
use_pretrained = False
pre_model_path = ""

# augmentation
# custom
# training
custom_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        # A.OneOf(
        #     [A.Resize(height=224, width=224), A.RandomCrop(height=224, width=224)], p=1
        # ),
        # A.RandomCrop(height=224, width=224),
        A.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),
        A.OneOf(
            [
                A.ElasticTransform(
                    p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ],
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        # A.Rotate(10),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)


# testdata
custom_test = A.Compose(
    [
        A.Resize(height=256, width=256),
         A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)


# torch transform
transforms_T = T.Compose(
    [T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5,), (0.5,)),]
)
