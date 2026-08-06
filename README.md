# Convolutions

Hands-on implementations of convolutional neural network building blocks, from raw
cross-correlation up to a full LeNet trained on FashionMNIST.

## Contents

### `notebooks/` — building blocks from scratch

| Notebook | What it covers |
| --- | --- |
| `conv.ipynb` | 2D cross-correlation (`corr2d`) and a hand-rolled `Conv2D` layer |
| `conv_channels.ipynb` | Multi-input/output channels and the 1×1 convolution as matrix multiplication |
| `pooling.ipynb` | Max/average pooling, with stride, padding, and multiple channels |
| `edge_detect.ipynb` | Edge detection with a Laplacian kernel, plus training a conv layer to learn it |

### `LeNet/` — a reusable model package

| File | Purpose |
| --- | --- |
| `model.py` | The `LeNet` module (lazy conv/linear layers) + Xavier `init_weights` helper |
| `train.py` | A typed `train()` loop |

### `LeNet.ipynb` — end-to-end pipeline

Loads FashionMNIST → trains LeNet (on MPS/CUDA/CPU) → predicts and visualizes a
batch → reports full test-set accuracy.

## Setup

```bash
uv sync
```

Then launch Jupyter against the project venv, e.g.:

```bash
uv run jupyter lab
```

## Running the LeNet example

Open `LeNet.ipynb` and run the cells top to bottom. The notebook:

1. Imports the model package (`from LeNet.model import LeNet`) — **run from the repo
   root** so the `LeNet` package is importable.
2. Downloads FashionMNIST to `./data/` on first run.
3. Selects a device automatically.
4. Trains, then predicts and evaluates.

### Notes on the model

- The conv/linear layers are **lazy** (`LazyConv2d`/`LazyLinear`), so their weights
  don't exist until the first forward pass. Call `model.apply_init((1, 1, 28, 28))`
  before creating the optimizer, it materializes the weights with a dummy batch and
  then applies Xavier initialization.
- LeNet uses **sigmoid** activations. It's sensitive to batch size and learning rate:
  large batches can saturate the sigmoids and stall training (loss stuck at
  `ln(10) ≈ 2.30`). A `batch_size` of 64–128 with SGD `lr≈0.1` reaches ~85% test
  accuracy in ~15 epochs.

## Type checking

The `LeNet` package is Typed. Check it with the Pyright:

```bash
uv run pyright LeNet
```

