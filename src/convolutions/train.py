import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def _sync(device: torch.device) -> None:
    # Block until queued GPU work finishes so timings are honest (dispatch is async).
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _mem_mb(device: torch.device) -> tuple[float, float]:
    # (live allocated MB, driver/reserved MB). MPS has no built-in peak counter,
    # so the caller tracks peak by sampling this each batch.
    if device.type == "mps":
        return (
            torch.mps.current_allocated_memory() / 1024**2,
            torch.mps.driver_allocated_memory() / 1024**2,
        )
    if device.type == "cuda":
        return (
            torch.cuda.memory_allocated() / 1024**2,
            torch.cuda.memory_reserved() / 1024**2,
        )
    return (0.0, 0.0)

def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: Optimizer,
    train_data_loader: DataLoader,
    epochs: int,
    device: torch.device,
    scheduler: LRScheduler | None = None,
) -> None:
    model.to(device)
    model.train()

    # Mixed precision on CUDA: autocast routes conv/matmul through Tensor Cores
    # (much faster than fp32). Prefer bf16 — it has fp32's exponent range, so it
    # can't overflow the way fp16 does (fp16 max ~65504 -> NaN on large early
    # activations) and needs no GradScaler. Fall back to fp16 + GradScaler on
    # GPUs without bf16. Disabled on MPS/CPU, so those devices are unaffected.
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    if use_amp:
        torch.backends.cudnn.benchmark = True  # autotune conv kernels for fixed input
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
    # scaling is only needed for fp16 (bf16's wide exponent range makes it moot)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    for epoch in range(epochs):
        epoch_loss: float = 0.0
        seen: int = 0
        peak_live: float = 0.0

        _sync(device)
        epoch_start = time.perf_counter()
        for (X_batch, y_batch) in train_data_loader:
            X, y = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()

            if use_amp:
                # CUDA: autocast + (fp16 only) gradient scaling
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    Y = model(X)
                    loss = loss_fn(Y, y).mean()
                epoch_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                # MPS/CPU: plain fp32 path — identical to the pre-AMP loop, so
                # no autocast/scaler overhead is added on these devices.
                Y = model(X)
                loss = loss_fn(Y, y).mean()
                epoch_loss += loss.item()
                loss.backward()
                optimiser.step()

            # throughput + memory profiling
            seen += X.size(0)
            live, _ = _mem_mb(device)
            peak_live = max(peak_live, live)
            # print(f">>> epoch : {epoch} | batch loss (mean over batch) : {loss.item()}")

        lr = optimiser.param_groups[0]["lr"]
        # step the lr schedule once per epoch (after the optimiser steps)
        if scheduler is not None:
            scheduler.step()

        _sync(device)
        elapsed = time.perf_counter() - epoch_start
        throughput = seen / elapsed
        live, driver = _mem_mb(device)
        print(
            f">>> epoch : {epoch} | epoch loss (train) : {epoch_loss / len(train_data_loader)} "
            f"| lr {lr:.4f} | {elapsed:.1f}s | {throughput:.1f} img/s "
            f"| mem live {live:.1f}MB peak {peak_live:.1f}MB driver {driver:.1f}MB >>>"
        )
