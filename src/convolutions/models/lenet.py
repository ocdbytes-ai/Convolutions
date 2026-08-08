import torch
import torch.nn as nn
from collections.abc import Sequence

def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # Lazy variants subclass these
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        channels_layer_1_conv = 6
        kernel_size_layer_1_conv = 5
        padding_layer_1_conv = 2

        kernel_size_pooling_layer_1 = 2
        stride_pooling_layer_1 = 2

        channels_layer_2_conv = 16
        kernel_size_layer_2_conv = 5

        kernel_size_pooling_layer_2 = 2
        stride_pooling_layer_2 = 2

        weights_linear_layer_1 = 120
        weights_linear_layer_2 = 84

        self.net = nn.Sequential(
            nn.LazyConv2d(channels_layer_1_conv, kernel_size=kernel_size_layer_1_conv, padding=padding_layer_1_conv),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_layer_1, stride=stride_pooling_layer_1),
            nn.LazyConv2d(channels_layer_2_conv, kernel_size=kernel_size_layer_2_conv),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=kernel_size_pooling_layer_2, stride=stride_pooling_layer_2),
            nn.Flatten(),
            nn.LazyLinear(weights_linear_layer_1), nn.Sigmoid(),
            nn.LazyLinear(weights_linear_layer_2), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.net(X)
        return Y

    # Lazy layers build their weights on first forward, so materialize
    # them with a dummy batch before applying Xavier init.
    def apply_init(self, X_shape: Sequence[int]) -> None:
        self.forward(torch.zeros(*X_shape))
        self.net.apply(init_weights)

    def layer_summary(self, X_shape: Sequence[int]) -> None:
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)