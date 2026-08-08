import torch
import torch.nn as nn
from collections.abc import Sequence

def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):  # Lazy variants subclass these
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Alex Net Model 
# 
# All the values for layer dimensions are taken from AlexNet paper
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            # Conv 1 (with ReLU normalization)
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), 
            nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 2 (with ReLU normalization)
            nn.LazyConv2d(256, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 3
            nn.LazyConv2d(384, kernel_size=3, padding=1), 
            nn.ReLU(),

            # Conv 4
            nn.LazyConv2d(384, kernel_size=3, padding=1), 
            nn.ReLU(),

            # conv 5
            nn.LazyConv2d(256, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Flatten(),

            nn.LazyLinear(4096), 
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.LazyLinear(4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.net(X)
        return Y

    # Lazy layers build their weights on first forward, so materialize
    # them with a dummy batch before applying Kaiming init.
    def apply_init(self, X_shape: Sequence[int]) -> None:
        self.forward(torch.zeros(*X_shape))
        self.net.apply(init_weights)

    def layer_summary(self, X_shape: Sequence[int]) -> None:
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)