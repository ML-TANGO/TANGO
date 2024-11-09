import torch
from torch import nn, Tensor
import torch.nn.functional as F


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 컨볼루션 레이어"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 컨볼루션 레이어 (패딩 포함)"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    """Bottleneck 블록 정의"""

    expansion: int = 4  # 확장 계수

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        # 1x1 컨볼루션 레이어
        self.conv1: nn.Conv2d = conv1x1(inplanes, planes)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        # 3x3 컨볼루션 레이어
        self.conv2: nn.Conv2d = conv3x3(planes, planes, stride)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        # 1x1 컨볼루션 레이어 (모델 변동사항)
        self.conv3: nn.Conv2d = conv1x1(planes, planes * self.expansion)
        self.bn3: nn.BatchNorm2d = nn.BatchNorm2d(planes * self.expansion)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.downsample: nn.Module = downsample
        self.stride: int = stride

    def forward(self, x: Tensor) -> Tensor:
        identity: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet152(nn.Module):
    """ResNet152 모델 정의"""

    def __init__(self, num_classes: int = 2) -> None:
        super(ResNet152, self).__init__()
        self.inplanes: int = 64

        # 초기 컨볼루션 레이어
        self.conv1: nn.Conv2d = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(self.inplanes)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 레이어
        self.layer1: nn.Sequential = self._make_layer(Bottleneck, 64, 3)
        self.layer2: nn.Sequential = self._make_layer(Bottleneck, 128, 8, stride=2)
        self.layer3: nn.Sequential = self._make_layer(Bottleneck, 256, 36, stride=2)
        self.layer4: nn.Sequential = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 평균 풀링 및 완전 연결 레이어
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: nn.Module, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """ResNet 레이어 생성 함수"""
        downsample: nn.Module = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: list = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # layer1.0.conv3 등 포함
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 모델 인스턴스 생성 예시
# model = ResNet152(num_classes=2)
