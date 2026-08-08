# Convolutions

Hands-on implementations of convolutional neural network building blocks, from raw
cross-correlation up to full LeNet, AlexNet, and ResNet18 architectures trained from
scratch.

## Papers Referred

- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Gradient-Based Learning Applied to Document Recognition (LeNet)](https://github.com/georgezoto/Convolutional-Neural-Networks/blob/master/Papers/1998%20LeNet-5%20GradientBased%20Learning%20Applied%20to%20Document%20Recognition%20-%20LeCun%2C%20Bottou%2C%20Bengio%2C%20Hanner.pdf)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)

## Project layout

```
convolutions/
├─ src/convolutions/          # installable package
│  ├─ models/
│  │  ├─ lenet.py             # LeNet   + Xavier  init_weights
│  │  ├─ alexnet.py           # AlexNet + Kaiming init_weights
│  │  └─ resnet.py            # ResNet18, Residual, ResNetInitParams
│  ├─ data.py                 # CIFAR-10 loaders (augmentation + normalization)
│  ├─ train.py                # training loop w/ throughput + memory profiling
│  └─ cli.py                  # `train-resnet` entry point (CUDA → MPS → CPU)
├─ scripts/train_resnet.py    # thin shim: python scripts/train_resnet.py
├─ notebooks/
│  ├─ concepts/               # conv, channels, pooling, edge-detection from scratch
│  └─ models/                 # LeNet.ipynb, ResNet18.ipynb end-to-end pipelines
├─ deploy/                    # Dockerfile + docker-compose.yml (CUDA training)
├─ data/                      # datasets (git-ignored, downloaded on first run)
└─ outputs/                   # checkpoints
```

## Contents

### `notebooks/concepts/` — building blocks from scratch

| Notebook | What it covers |
| --- | --- |
| `conv.ipynb` | 2D cross-correlation (`corr2d`) and a hand-rolled `Conv2D` layer |
| `conv_channels.ipynb` | Multi-input/output channels and the 1×1 convolution as matrix multiplication |
| `pooling.ipynb` | Max/average pooling, with stride, padding, and multiple channels |
| `edge_detect.ipynb` | Edge detection with a Laplacian kernel, plus training a conv layer to learn it |

### `notebooks/models/` — end-to-end pipelines

| Notebook | Dataset | What it does |
| --- | --- | --- |
| `LeNet.ipynb` | FashionMNIST | Trains LeNet → predicts and visualizes a batch → full test-set accuracy |
| `ResNet18.ipynb` | CIFAR-10 | Trains ResNet18 (cosine LR) with throughput/memory profiling → top-1/top-5 accuracy |

### `src/convolutions/models/`

| Module | Purpose |
| --- | --- |
| `lenet.py` | `LeNet` module (lazy conv/linear layers) + Xavier `init_weights` |
| `alexnet.py` | `AlexNet` module (lazy conv/linear layers) + Kaiming `init_weights` |
| `resnet.py` | `ResNet18`, the `Residual` block, and `ResNetInitParams` config |

## Setup

Installs dependencies **and** the `convolutions` package (editable, src-layout) so
`from convolutions.models.resnet import ResNet18` works from anywhere:

```bash
uv sync
```

Launch Jupyter against the project venv:

```bash
uv run jupyter lab
```

The model notebooks live in `notebooks/models/` and resolve `data/` via a `../../data`
relative path, so run them from their own directory (Jupyter's default).

## Training ResNet18 on CIFAR-10

### Locally

```bash
uv run train-resnet --epochs 100 --batch-size 128 --workers 4
# or, equivalently:
uv run python scripts/train_resnet.py --epochs 100
```

The entry point selects a device automatically (**CUDA → MPS → CPU**), trains with a
`CosineAnnealingLR` schedule, reports per-epoch throughput (img/s) and memory, prints
top-1/top-5 test accuracy, and saves a checkpoint to `outputs/resnet18_cifar10.pt`.

Flags (all have env-var equivalents, e.g. `EPOCHS`, `BATCH_SIZE`, `LR`, `WORKERS`):
`--epochs`, `--batch-size`, `--lr`, `--workers`, `--data-root`, `--out-dir`.

### On a CUDA machine via Docker (e.g. Ubuntu on WSL)

Prerequisites: NVIDIA driver installed **on Windows**, and `nvidia-container-toolkit`
inside the WSL distro. Run `nvidia-smi` on the host and check the **CUDA Version** it
reports (top-right) is **≥ 13.0** that's the max CUDA your driver supports, and this
project's torch needs CUDA 13.

> First time setting up the GPU, or hitting `could not select device driver "nvidia"`?
> See **[deploy/cuda_setup.md](deploy/cuda_setup.md)** for step-by-step WSL and native
> Linux setup + troubleshooting.

Build the image once (via compose, so the tag/context match the training run), then
confirm the container actually sees the GPU using the project's own CUDA-13 torch:

```bash
docker compose -f deploy/docker-compose.yml build
docker run --rm --gpus all --entrypoint python resnet18-cifar10:latest \
  -c "import torch; print('cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

If that prints `cuda available: True` with your GPU name, start training with the
already-built image:

```bash
docker compose -f deploy/docker-compose.yml up
```

This mounts `./data` (CIFAR persists across runs) and `./outputs` (checkpoint), and
raises `/dev/shm` to 2 GB so multi-worker DataLoaders don't hit bus errors.

## Notes on the models

- **Lazy layers.** Conv/linear layers are `LazyConv2d`/`LazyLinear`, so their weights
  don't exist until the first forward pass. Call `model.apply_init(...)` before creating
  the optimizer — it materializes the weights with a dummy batch (in eval mode, so
  BatchNorm doesn't choke on a size-1 batch) and then applies initialization.
- **LeNet** uses **sigmoid** activations and is sensitive to batch size / learning rate:
  large batches can saturate the sigmoids and stall training (loss stuck at
  `ln(10) ≈ 2.30`). `batch_size` 64–128 with SGD `lr≈0.1` reaches ~85% test accuracy in
  ~15 epochs.
- **ResNet18** uses the **CIFAR stem** (3×3 stride-1 conv, no maxpool) rather than the
  ImageNet stem (7×7 stride-2 + maxpool), which would collapse a 32×32 image to 1×1
  before the last stage. Trained with cosine LR decay over ~50–100+ epochs.

## Type checking

The package is typed. Check it with Pyright:

```bash
uv run pyright src/convolutions
```
