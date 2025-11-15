import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import PolypDataset, get_train_transform, get_val_transform
from model import build_unetpp_model, HybridLoss

import segmentation_models_pytorch as smp


# ----------------------------
#       CONFIGS
# ----------------------------

DATA_PATH = "data"
IMAGE_DIR = f"{DATA_PATH}/images"
MASK_DIR = f"{DATA_PATH}/masks"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"


# ----------------------------
#       METRICS
# ----------------------------

def dice_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() + 1e-7
    return (2.0 * intersection) / union


# ----------------------------
#       TRAINING LOOP
# ----------------------------

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for img, mask in tqdm(loader, total=len(loader), desc="Training"):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        preds = model(img)
        loss = loss_fn(preds, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ----------------------------
#       VALIDATION LOOP
# ----------------------------

def val_fn(loader, model):
    model.eval()
    total_dice = 0

    with torch.no_grad():
        for img, mask in tqdm(loader, total=len(loader), desc="Validating"):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            preds = model(img)
            total_dice += dice_score(preds, mask).item()

    return total_dice / len(loader)


# ----------------------------
#       MAIN TRAIN SCRIPT
# ----------------------------

def main():
    print(f"Using device: {DEVICE}")

    # Dataset
    train_dataset = PolypDataset(
        IMAGE_DIR,
        MASK_DIR,
        transform=get_train_transform()
    )

    val_dataset = PolypDataset(
        IMAGE_DIR,
        MASK_DIR,
        transform=get_val_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = build_unetpp_model().to(DEVICE)

    # Loss + Optimizer
    loss_fn = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice = 0

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch+1}/{EPOCHS}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        dice = val_fn(val_loader, model)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice Score: {dice:.4f}")

        # Save best model
        if dice > best_dice:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            best_dice = dice
            print("Saved new best model!")

    print("\nTraining Finished!")
    print(f"Best Dice Score: {best_dice:.4f}")


if __name__ == "__main__":
    main()
