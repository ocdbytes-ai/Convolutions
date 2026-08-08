"""Dataset / DataLoader factories."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Per-channel CIFAR-10 training-set statistics (R, G, B).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Train/test CIFAR-10 loaders with standard augmentation + normalization.
def cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    persistent = workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, persistent_workers=persistent, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, persistent_workers=persistent, pin_memory=pin_memory,
    )
    return train_loader, test_loader
