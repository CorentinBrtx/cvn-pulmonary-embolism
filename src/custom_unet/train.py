"""Train our U-Net model"""

import os
from logging import Logger
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.utils.logger import SummaryWriter


def evaluate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for _, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)
    return val_loss


def diceBCELoss(inputs: torch.TensorType, targets: torch.TensorType, smooth=1):
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
    Dice_BCE = BCE + dice_loss

    return Dice_BCE


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss,
    logger: Logger,
    scheduler: torch.optim.lr_scheduler = None,
    num_epochs: int = 1000,
    device: Literal["cpu", "cuda"] = "cuda",
    model_path: str = "",
):
    """Train the model"""

    if model_path:
        os.makedirs(model_path, exist_ok=True)

    train_losses = []
    val_losses = []

    summary_writer = SummaryWriter(num_epochs, len(train_loader), len(val_loader), logger)

    for epoch in range(num_epochs):

        summary_writer.set_epoch(epoch)

        # Training
        model.train()
        model.to(device)
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                summary_writer("train", i, {"loss": loss.item()})

        epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_loss)

        logger.info(f"Train Epoch Loss: {epoch_loss:.4f}\n")

        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)

        val_losses.append(val_loss)

        logger.info(f"Validation Epoch Loss: {val_loss:.4f}\n")

        if scheduler is not None:
            scheduler.step()

        # Save the model
        if model_path:
            torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))

        fig, ax1 = plt.subplots(figsize=(12, 7), dpi=200, facecolor="w")
        ax2 = ax1.twinx()
        ax1.plot(train_losses, label="loss train", color="blue")
        ax1.plot(val_losses, label="loss validation", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.legend()
        ax2.legend(loc="upper center")
        fig.savefig(os.path.join(model_path, "progress.png"))
        plt.close(fig)

    return train_losses, val_losses
