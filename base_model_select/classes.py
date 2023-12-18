import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

class DataLoaderWrapper:
    def set_train_loader(self, train_dataset):
        self._train_data_loader = DataLoader(train_dataset)

    def set_test_loader(self, test_dataset):
        self._test_data_loader = DataLoader(test_dataset)

    def get_train_loader(self):
        return self._train_data_loader

    def get_test_loader(self):
        return self._test_data_loader


class PandasDataset(Dataset):
    def __init__(self, df, num_classes):
        self._df = df
        self._num_classes = num_classes

    @property
    def feat_size(self):
        feat = self._df["feat"].to_numpy()[0]
        #print(feat.shape)
        # print("Shape: ", np.prod(feat.shape))
        return np.prod(feat.shape)
        # return 1

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        feat, label = (
            self._df["feat"].to_numpy()[index],
            self._df["label"].to_numpy()[index],
        )
        label_np = np.zeros(self._num_classes)
        label_np[label] = 1
        return torch.flatten(torch.from_numpy(feat)), torch.flatten(torch.from_numpy(label_np))
    
    
class PandasTest(Dataset):
    def __init__(self, df, num_classes):
        self._df = df
        self._num_classes = num_classes

    @property
    def feat_size(self):
        feat = self._df["feat"].to_numpy()[0]
        #print(feat.shape)
        # print("Shape: ", np.prod(feat.shape))
        return np.prod(feat.shape)
        # return 1

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        feat = self._df["feat"].to_numpy()[index]
        path = self._df["path"].to_numpy()[index]

        # label_np = np.zeros(self._num_classes)
        # label_np[label] = 1
        return torch.from_numpy(feat), path

class VOCDetection:
    def __init__(self, *args, **kwargs):
        import torchvision
        self._dataset = torchvision.datasets.VOCDetection(**kwargs)
        self.size = len(self._dataset.images)
        
    def set_size(self, size):
        if size > len(self._dataset.images):
            raise ValueError("Size cannot be greater than the number of images in the dataset")
        self.size = size
        

    def __len__(self):
        return self.size
        # return 10 #We are looking at a max of 10 images

    def __getitem__(self, index: int) -> str:
        return self._dataset.images[index]
    
class COCO:
    def __init__(self, *args, **kwargs):
        import torchvision
        self._dataset = torchvision.datasets.CocoDetection(**kwargs)
        # self.size = len(self._dataset.images)
        
    def set_size(self, size):
        if size > len(self._dataset.images):
            raise ValueError("Size cannot be greater than the number of images in the dataset")
        self.size = size
        

    def __len__(self):
        return self.size
        # return 10 #We are looking at a max of 10 images

    # def __getitem__(self, index: int) -> str:
    #     return self._dataset.images[index]