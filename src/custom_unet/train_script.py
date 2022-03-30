import os
from argparse import ArgumentParser

import torch

from src.custom_unet.custom_unet import CustomUNet
from src.custom_unet.dataloaders import get_data_loaders
from src.custom_unet.train import diceBCELoss, evaluate, train
from src.utils.logger import setup_logger

parser = ArgumentParser()

parser.add_argument("--train_img_path", type=str)
parser.add_argument("--train_seg_path", type=str)
parser.add_argument("--test_img_path", type=str, default="")
parser.add_argument("--test_seg_path", type=str, default="")
parser.add_argument("--model_path", type=str, default="/gpfs/workdir/shared/pulmembol/custom_unet")

args = parser.parse_args()

model_dir = args.model_path

os.makedirs(model_dir, exist_ok=True)

logger = setup_logger(os.path.join(model_dir, "train.log"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {DEVICE}")

train_loader, val_loader, test_loader = get_data_loaders(
    args.train_img_path, args.train_seg_path, args.test_img_path, args.test_seg_path
)
model = CustomUNet()
criterion = diceBCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=None,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=1000,
    device=DEVICE,
    model_path=args.model_path,
    logger=logger,
)

""" # Test
test_loss = evaluate(model, test_loader, criterion, "cuda")
print(f"Mean loss: {test_loss:.3f}") """
