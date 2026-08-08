# syntax=docker/dockerfile:1
#
# ResNet18 / CIFAR-10 training image — installs the EXACT versions the project
# pins (torch 2.13.0 / torchvision 0.28.0) via uv + uv.lock
FROM python:3.14-slim-bookworm

# uv binary, copied from its published image (robust across base images).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependency layer — cached until pyproject.toml / uv.lock change.
#    --no-install-project installs third-party deps (torch, …)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project

# 2) Project source, then install our package into the same venv.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# venv on PATH so the `train-resnet` console script runs directly.
ENV PATH="/app/.venv/bin:$PATH"

# Fail the build early if torch didn't resolve to a CUDA build.
RUN python -c "import torch, torchvision; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

# argparse flags (e.g. --epochs 100) append after the image name.
# data/ and outputs/ are mounted at runtime so the CIFAR download persists.
ENTRYPOINT ["train-resnet"]
