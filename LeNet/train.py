import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: Optimizer,
    train_data_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss: float = 0.0
        for (X_batch, y_batch) in train_data_loader:
            X, y = X_batch.to(device), y_batch.to(device)
            Y = model(X)
            loss = loss_fn(Y, y).mean()
            epoch_loss += loss.item()

            # backpropogation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            # print(f">>> epoch : {epoch} | batch loss (mean over batch) : {loss.item()}")
        print(f">>> epoch : {epoch} | epoch loss (train) : {epoch_loss / len(train_data_loader)} >>>")
