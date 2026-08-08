# Train ResNet18 on CIFAR-10. Prefers CUDA, falls back to MPS then CPU.
# Console entry point (`train-resnet`) and the target of scripts/train_resnet.py.


import argparse
import os

import torch

from convolutions.data import cifar10_loaders
from convolutions.models.resnet import ResNet18, ResNetInitParams
from convolutions.train import train


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 50)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 128)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 0.1)))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./data"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./outputs"))
    args = parser.parse_args()

    device = get_device()
    print(f"using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, test_loader = cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(f"train: {len(train_loader.dataset)} | test: {len(test_loader.dataset)}", flush=True)

    model = ResNet18(ResNetInitParams(
        arch=[(2, 64), (2, 128), (2, 256), (2, 512)],
        num_classes=10,
    ))
    model.apply_init((1, 3, 32, 32))
    model = model.to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    train(model, loss_fn, optimiser, train_loader,
          epochs=args.epochs, device=device, scheduler=scheduler)

    # top-1 / top-5 on the full test set
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X, y = X_batch.to(device), y_batch.to(device)
            _, top5_idx = model(X).topk(5, dim=1)
            correct = top5_idx == y.unsqueeze(1)
            top1 += correct[:, 0].sum().item()
            top5 += correct.any(dim=1).sum().item()
            total += y.size(0)
    print(f"top-1: {top1 / total:.2%} ({top1}/{total})", flush=True)
    print(f"top-5: {top5 / total:.2%} ({top5}/{total})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "resnet18_cifar10.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
