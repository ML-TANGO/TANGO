import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from collections import OrderedDict
from typing import List


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        # 첫 번째 Batch Normalization 레이어
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        # 첫 번째 ReLU 활성화 함수
        self.add_module("relu1", nn.ReLU(inplace=True))
        # 1x1 Convolution 레이어
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        # 두 번째 Batch Normalization 레이어
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        # 두 번째 ReLU 활성화 함수
        self.add_module("relu2", nn.ReLU(inplace=True))
        # 3x3 Convolution 레이어
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate: float = drop_rate
        self.memory_efficient: bool = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        # 입력 텐서들을 채널 방향으로 연결
        concated_features: Tensor = torch.cat(inputs, 1)
        # BatchNorm -> ReLU -> Conv1 연산 수행
        bottleneck_output: Tensor = self.conv1(
            self.relu1(self.norm1(concated_features))
        )
        return bottleneck_output

    def forward(self, input: List[Tensor]) -> Tensor:
        if self.memory_efficient and any(tensor.requires_grad for tensor in input):
            # 메모리 효율적 모드에서 체크포인트 사용
            bottleneck_output: Tensor = cp.checkpoint(self.bn_function, input)
        else:
            bottleneck_output = self.bn_function(input)

        new_features: Tensor = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                memory_efficient,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features: Tensor) -> Tensor:
        # 초기 특징 맵 리스트
        features: List[Tensor] = [init_features]
        for name, layer in self.named_children():
            new_features: Tensor = layer(features)
            features.append(new_features)
        # 모든 특징 맵들을 채널 방향으로 연결
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        # BatchNorm -> ReLU -> Conv -> AvgPool 레이어로 구성
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet201(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        drop_rate: float = 0,
        memory_efficient: bool = False,
    ) -> None:
        super(DenseNet201, self).__init__()
        growth_rate: int = 32
        block_config: tuple = (6, 12, 48, 32)
        num_init_features: int = 64
        bn_size: int = 4

        # 첫 번째 컨볼루션 레이어: 입력 채널을 1로 설정
        self.features: nn.Sequential = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            1,  # 입력 채널 수를 1로 설정
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # DenseBlock과 Transition 레이어들 생성
        num_features: int = num_init_features
        for i, num_layers in enumerate(block_config):
            # DenseBlock 추가
            block = _DenseBlock(
                num_layers,
                num_features,
                bn_size,
                growth_rate,
                drop_rate,
                memory_efficient,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                # Transition 레이어 추가
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        # 마지막 BatchNorm 레이어 추가
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # 분류기 레이어
        self.classifier: nn.Linear = nn.Linear(num_features, num_classes)

        # 마지막에 conv1 레이어 추가
        self.conv1 = nn.Conv2d(
            1,  # 입력 채널 수
            num_init_features,  # 출력 채널 수 (보통 64)
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # 모델 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming He 초기화 사용
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BatchNorm 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 선형 레이어 초기화
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # 특징 추출
        features: Tensor = self.features(x)
        # ReLU 활성화 함수 적용
        out: Tensor = F.relu(features, inplace=True)
        # Adaptive Average Pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # 텐서 펼치기
        out = torch.flatten(out, 1)
        # 분류기 통과
        out = self.classifier(out)
        return out
