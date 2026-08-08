import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from collections.abc import Sequence

def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # Lazy variants subclass these
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Residual(nn.Module):
    # num_channels : output channels 
    def __init__(self, num_channels:int, use_1x1_conv_layer:bool=False, strides:int=1) -> None:
        super().__init__()
        self.conv1 = nn.LazyConv2d(
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides
        )
        self.conv2 = nn.LazyConv2d(
            num_channels,
            kernel_size=3,
            padding=1,
        )

        # correction layer (if we want to change the number of channels)
        # this is the W_s projection logic given in the paper
        if use_1x1_conv_layer:
            self.conv3 = nn.LazyConv2d(
                num_channels,
                kernel_size=1,
                stride=strides
            )
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

@dataclass
class ResNetInitParams:
    arch: list[Sequence[int]]
    num_classes: int

# 18 Layered 
# ResNet
class ResNet18(nn.Module):
    def __init__(self, init_params: ResNetInitParams) -> None:
        super().__init__()
        self.net = nn.Sequential(self.block_1())
        for i, b in enumerate(init_params.arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module(
            'last linear block',
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.LazyLinear(init_params.num_classes)
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def block_1(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )

    def block(self, num_residual_layers:int, num_channels: int, first_block:bool=False) -> nn.Sequential:
        blocks = []
        for i in range(num_residual_layers):
            if i == 0 and not first_block:
                blocks.append(
                    Residual(num_channels, use_1x1_conv_layer=True, strides=2)
                )
            else:
                blocks.append(
                    Residual(num_channels)
                )
        return nn.Sequential(*blocks)

    # Lazy layers build their weights on first forward, so materialize
    # them with a dummy batch before applying Kaiming init. Run in eval mode:
    # a training-mode forward would make BatchNorm compute per-batch variance,
    # which needs >1 value per channel (a [1, C, 1, 1] feature map fails), and
    # would also pollute BN's running stats with these dummy zeros.
    def apply_init(self, X_shape: Sequence[int]) -> None:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(torch.zeros(*X_shape))
        self.train(was_training)
        self.net.apply(init_weights)

    def layer_summary(self, X_shape: Sequence[int]) -> None:
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
