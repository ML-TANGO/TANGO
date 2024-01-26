import os
import time
import torch

from torch import nn
from torch import optim
from dataclasses import dataclass

from PIL import Image
import numpy as np


def entropy(path, x=3, y=3):  
    img = Image.open(path)
    yPieces = x
    xPieces = y
    
    entropies = []
    imgwidth, imgheight = img.size
    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            section = img.crop(box)
            entropies.append(section.entropy())

    # Output must be a numpy array
    return np.array(entropies, dtype=np.float32)




class SVMPredictor():
    def __init__(self, input_size=None, output_size=None, pt=""):
        super(SVMPredictor, self).__init__()
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if pt == "":
            self._model = nn.Linear(input_size, output_size)
        else:
            # pt 
            self._model = nn.Linear(input_size, output_size)
            self._model.load_state_dict(torch.load(pt, map_location=self._device))
            
        self.sizes = (input_size, output_size)

        self.stats = [0 for x in range(output_size)]
        
        self._model.to(self._device)

        self._best_acc = 0
        self._name = "SVM"
        
    def forward(self, x):
        x = self._model(x)
        return x
        # return torch.argmax(x).item()
        
    def _train(self, data_loader):
        self._model.train()

        train_loss, correct, total = 0, 0, 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self._model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
        )
        
        first = False
        
        for batch_idx, (inputs, targets) in enumerate(data_loader.get_train_loader()):
            
            # Moves tensor to device GPU/CPU
            inputs, targets = inputs.to(self._device), targets.to(self._device) 

            outputs = self._model(inputs)       # Forward Pass
            loss = criterion(outputs, targets)  # Calculate loss
            
            optimizer.zero_grad()               # Optimize
            loss.backward()
            optimizer.step()
            
            
            train_loss += loss.item()
            
            # Compare predicted and actual results
            predicted = (torch.argmax(outputs[0])).item()
            actual = (torch.argmax(targets[0])).item()

            # print(predicted, actual)
            
            # Used to calculate accuracy                        
            total += 1
            correct += (predicted == actual)            

    def _test(self, data_loader):
        self._model.eval()

        test_loss, correct, total = 0, 0, 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader.get_test_loader()):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs)
                outputs = self._model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()

                # print(outputs)
                # print(outputs[0])
                
                predicted = (torch.argmax(outputs[0])).item()
                actual = (torch.argmax(targets[0])).item()

                self.stats[predicted] += 1
            
                total += 1
                correct += (predicted >= actual)

        acc = 100.0 * correct / total
        if acc > self._best_acc:
            # print("Saving..", acc)
            state = {
                "model": self._model.state_dict(),
                "acc": acc,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/{}_ckpt.pth".format(self._name))
            self._best_acc = acc
            
        return correct == total

    def train(self, data_loader, epoch=5):
        for e in range(epoch):
            self._train(data_loader)
            if e % 5 == 0:
                self._test(data_loader)
                # break
                # print(f"Early stopping at epoch {e}")

        self.print_stats()
        return self._best_acc
        
    def parse(self, path):
        import xml.etree.ElementTree as ET

        # parse xml file
        tree = ET.parse(path) 
        root = tree.getroot() # get root object
        
        height = int(root.find("size")[0].text)
        width = int(root.find("size")[1].text)
        channels = int(root.find("size")[2].text)
        
        boxes = []
        labels = []
        
        names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        named = {}
        for i, name in enumerate(names):
            named[name] = i
        
        for member in root.findall('object'):
            class_name = member[0].text # class name
                
            if class_name not in named:
                continue
            
            # bbox coordinates
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            # store data in list
            labels.append(named[class_name])
            boxes.append([xmin, ymin, xmax, ymax])

        return [{
            "boxes": torch.FloatTensor(boxes),
            "labels": torch.as_tensor(labels),
        }]
    
    def process_data(self, data_loader):
        df = pd.DataFrame()
        append_data = []
        for img_path_list in data_loader:
            img_path = img_path_list[0]                 # Load an image
            append_data.append({
                "path": img_path,
                "feat": selector(img_path)
            })
        df = pd.concat([df, pd.DataFrame(append_data)], ignore_index=True)
        return PandasTest(df, len(predictor_list))
    
    def pad(self, data_loader):
        df = pd.DataFrame()
        append_data = []
        for img_path_list in data_loader:
            img_path = img_path_list[0]                 # Load an image
            append_data.append({
                "path": img_path,
                "feat": np.zeros(6)
            })
        df = pd.concat([df, pd.DataFrame(append_data)], ignore_index=True)
        return PandasTest(df, len(predictor_list))

    @torch.no_grad()
    def test(self, test_dataset, bias=None, random=False):
        self._model.eval()
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        map = MeanAveragePrecision(box_format="xyxy")
        
        t1 = time.process_time()
        if bias is None:
            test_set = self.process_data(DataLoader(test_dataset, shuffle=True))
        else:
            test_set = self.pad(DataLoader(test_dataset, shuffle=True))
            
        t2 = time.process_time()
        print(f"Pre-processing time: {(t2-t1):.3f}")

        
        # feature_extractor = ResNet18Extractor() #Auto puts into CUDA

        t1 = time.process_time()

        for batch_idx, (feat, img_path) in enumerate(DataLoader(test_set)):
            
            path = img_path[0]
            
            
            if bias is None:
                feat = feat.to(self._device)                
                outputs = self._model(feat)       # Forward Pass            
                # Compare predicted and actual results
                pred_indx = (torch.argmax(outputs[0])).item()
            elif random:
                pred_indx = random.randint(0, len(predictor_list)-1)
                print(pred_indx)
            else:
                pred_indx = bias
                
            result = predictor_list[pred_indx].detect_(path)
            out = result.pandas().xyxy[0]
            # print(out)
            if out.empty:
                preds = [
                    {
                        "boxes": torch.FloatTensor([]),
                        "scores": torch.FloatTensor([]),
                        "labels": torch.FloatTensor([]),
                    }
                ]
            else:
                preds = [
                    {
                        "boxes": torch.FloatTensor(out.iloc[:, :4].values),
                        "scores": torch.FloatTensor(out.loc[:, "confidence"].values),
                        "labels": torch.tensor(out.loc[:, "class"].values),
                    }
                ]

            # preds = self.get_preds(result.pandas().xyxy[0])
            # print(preds)
            
            path = Path(path)
            ann_path = path.parent.parent / "Annotations" / (path.name.split(".")[0] + ".xml")
            # print(ann_path)
            if ann_path.exists():
                targets = self.parse(ann_path)
                map.update(preds, targets)
            else:
                print(f"{str(ann_path)} was not found")
                    
        t2 = time.process_time()
        print(f"Inference time: {(t2-t1):.3f}")
            
        print(f"mAP: {map.compute()['map'].item():.3f}")

        return map.compute()['map'].item()

    def print_stats(self):
        total = 0
        print("Model selection percentages")
        for i in range(len(self.stats)):
            total += self.stats[i]

        for i in range(len(self.stats)):
            print(f"{i}: {self.stats[i]/total}")   
        
        print(f"{self._best_acc:.2f}\n")
