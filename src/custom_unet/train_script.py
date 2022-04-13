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
parser.add_argument("--log_path", type=str)
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--internal_channels", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--iterable", action="store_true")
parser.add_argument("--patch_size", type=lambda s: [int(i) for i in s.split(",")], default=[64, 64, 64])

args = parser.parse_args()

model_dir = args.model_path

os.makedirs(model_dir, exist_ok=True)

NUM_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK", 0))

log_path = args.log_path or os.path.join(model_dir, "train.log")
logger = setup_logger(log_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {DEVICE}")

train_loader, val_loader, test_loader = get_data_loaders(
    args.train_img_path, args.train_seg_path, args.test_img_path, args.test_seg_path, batch_size=args.batch_size, num_workers=NUM_WORKERS, patch_size=args.patch_size, iterable=args.iterable
)
model = CustomUNet(internal_channels=args.internal_channels, n_layers=args.n_layers)
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
    num_epochs=args.n_epochs,
    device=DEVICE,
    model_path=args.model_path,
    logger=logger,
)

# Test
# test_loss = evaluate(model, test_loader, criterion, "cuda")
# print(f"Mean loss on test set: {test_loss:.3f}") 
