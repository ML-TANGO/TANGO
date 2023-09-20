DEFAULT_OPTIONS = {
    "input_size": [256, 256],
    "input_mean": [0.5],
    "input_std": [0.5],
    "network_type": ["FP32"],  # FP32, FP16
    "epochs": [3],
    "optimizer": ["SGD", "NAG"],
    "initial_lr": [0.01, 0.005],
    "loss_function": ["CE"],
}

from pathlib import Path

import yaml
import torch
from torch import optim
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

from .models.resnet_cifar10 import ResNet, BasicBlock

from typing import Dict, List, Tuple


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


def get_optimizer(optimizer_name: str, model: nn.Module, lr=0.1, momentum=0.9) -> optim.Optimizer:
    optimizer_case = {
        "SGD": lambda lr, mo: optim.SGD(model.parameters(), lr=lr, momentum=mo, weight_decay=1e-4),
        "Adam": lambda lr, mo: optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4),
        "NAG": lambda lr, mo: optim.SGD(model.parameters(), lr=lr, momentum=mo, nesterov=True, weight_decay=5e-4),
        "RMSprop": lambda lr, mo: optim.RMSprop(model.parameters(), lr=lr),
    }.get(optimizer_name, None)
    if optimizer_case is None:
        raise ValueError("Invalid optimizer name: {}".format(optimizer_name))
    return optimizer_case(lr, momentum)


def get_loss_function(loss_function_name: str) -> nn.Module:
    loss_function_case = {
        "CE": nn.CrossEntropyLoss(label_smoothing=0.1),
        "FL": FocalLoss(),
    }.get(loss_function_name, None)
    if loss_function_case is None:
        raise ValueError("Invalid loss function name: {}".format(loss_function_name))
    return loss_function_case


def get_transforms(input_size: int, input_mean: float, input_std: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=input_mean, std=input_std),
        ]
    )


def validate_model(model: nn.Module, val_data_loader: DataLoader, device: torch.device, loss_function: nn.Module) -> Tuple[float, float]:
    model.to(device).eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_data_loader):
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            val_loss += loss.item()
            _, predicted = output.max(1)
            val_acc += predicted.eq(target.view_as(predicted)).sum().item()
    val_loss /= len(val_data_loader)
    val_acc /= len(val_data_loader.dataset)
    return val_loss, val_acc


def run_resnet(proj_path: str, dataset_yaml_path: str):
    proj_path = Path(proj_path)
    data_path = Path(dataset_yaml_path).parent
    last_pt = proj_path / "weights" / "last.pt"
    best_pt = proj_path / "weights" / "best.pt"
    if not last_pt.parent.exists():
        last_pt.parent.mkdir(parents=True, exist_ok=True)
    if not best_pt.parent.exists():
        best_pt.parent.mkdir(parents=True, exist_ok=True)


    with open(proj_path / "project_info.yaml", "r") as f:
        project_info: dict = yaml.safe_load(f)  #

    with open(proj_path / "basemodel.yaml", "r") as f:
        basemodel_info: dict = yaml.safe_load(f)


    device = torch.device("cuda" if torch.cuda.is_available() and project_info.get("acc", "cpu") == "cuda" else "cpu")
    batch_size: int = project_info.get("batch_size", 3)

    layers: List[int] = basemodel_info.get("layers", [3, 3, 3])
    num_classes: int = basemodel_info.get("num_classes", 2)
    input_size: int = DEFAULT_OPTIONS.get("input_size", [256])[0]
    input_mean: float = DEFAULT_OPTIONS.get("input_mean", [0.5])[0]
    input_std: float = DEFAULT_OPTIONS.get("input_std", [0.5])[0]
    network_type: str = DEFAULT_OPTIONS.get("network_type", ["FP32"])[0]
    epochs: int = DEFAULT_OPTIONS.get("epochs", [100])[0]
    optim_name: str = DEFAULT_OPTIONS.get("optimizer", ["SGD"])[0]
    lr: float = DEFAULT_OPTIONS.get("initial_lr", [0.01])[0]
    loss_name: str = DEFAULT_OPTIONS.get("loss_function", ["CE"])[0]

    # 데이터셋 / 데이터 로더 생성
    transform = get_transforms(input_size, input_mean, input_std)
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 모델 생성
    net = ResNet(BasicBlock, layers, num_classes).to(device)
    print(f"create model: resnet{sum(layers) * 2 + 2}")

    # 옵티마이저 생성
    optimizer: optim.Optimizer = get_optimizer(optim_name, net, lr=lr)
    print(f"create optimizer: {optim_name}")

    # 손실함수 생성
    loss_function: nn.Module = get_loss_function(loss_name)
    print(f"create loss function: {loss_name}")

    # 학습 시작
    print("start training")
    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (data, target) in enumerate(train_data_loader):
            scaler = amp.GradScaler()
            data = data.float().to(device)
            target = target.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                output = net(data)
                loss = loss_function(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_acc += predicted.eq(target.view_as(predicted)).sum().item()
        train_loss /= len(train_data_loader)
        train_acc /= len(train_data_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} train loss: {train_loss:.4f} train acc: {train_acc:.4f}")
        val_loss, val_acc = validate_model(net, val_data_loader, device, loss_function)
        print(f"Epoch {epoch + 1}/{epochs} val loss: {val_loss:.4f} val acc: {val_acc:.4f}\n")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), best_pt)
            print(f"save best model: {best_pt}")
        torch.save(net.state_dict(), last_pt)
        print(f"save last model: {last_pt}")

    return best_pt
