import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import prepare_dataloaders
from unet_model import UNet
from metrics import dice_loss, iou_score

def train_model(data_dir, epochs=50, batch_size=4, lr=1e-4, device='cuda', log_dir='runs/unet'):

    train_loader, val_loader = prepare_dataloaders(data_dir, batch_size)
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_iou = 0.0

    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            image, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
            if fov is not None:
                fov = fov.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = dice_loss(output, mask, fov)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        dice_scores, ious = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                image, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                if fov is not None:
                    fov = fov.to(device)

                output = model(image)
                dice = 1 - dice_loss(output, mask, fov)
                iou = iou_score(output, mask, fov)
                dice_scores.append(dice.item())
                ious.append(iou.item())

        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(ious) / len(ious)

        writer.add_scalar("Dice/val", avg_dice, epoch)
        writer.add_scalar("IoU/val", avg_iou, epoch)

        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f} | Dice={avg_dice:.4f} | IoU={avg_iou:.4f}")

        # Save best model
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), f"checkpoints/unet_best.pth")
            print("✅ Saved new best model.")

    writer.close()
