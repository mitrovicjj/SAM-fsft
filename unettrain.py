import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from unetmodel import UNet
from datasets import get_transforms, RetinaDataset
from utils import prepare_dataloaders
from metrics import dice_loss, iou_score
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.checkpoint import checkpoint_sequential

def train_model(data_dir, epochs=50, batch_size=4, lr=1e-4, device='cuda', log_dir='runs/unet'):
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    best_iou = 0.0
    base_image_size = 1024
    accumulation_steps = 4
    os.makedirs("checkpoints", exist_ok=True)

    def log_images(images, masks, preds, epoch, tag="Validation"):
        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()
        grid = make_grid(torch.cat([images, masks, preds], dim=0), nrow=images.size(0))
        writer.add_image(f"{tag}/image-mask-pred", grid, epoch)

    def attempt_training(batch_size, image_size):
        print(f"\n Starting training: batch_size={batch_size}, image_size={image_size}")
        transform = get_transforms(image_size)
        train_loader, val_loader = prepare_dataloaders(data_dir, batch_size, transform)

        model = UNet(n_channels=3, n_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"\n🔁 Epoch {epoch+1}/{epochs}")
            model.train()
            epoch_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                image, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                if fov is not None:
                    fov = fov.to(device)

                with torch.cuda.amp.autocast():
                    output = model(image)
                    loss = dice_loss(output, mask, fov) / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps

            avg_train_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation
            model.eval()
            dice_scores = []
            ious = []
            vis_logged = False
            pred_dir = f"predictions/epoch_{epoch+1}"
            os.makedirs(pred_dir, exist_ok=True)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                    image, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                    if fov is not None:
                        fov = fov.to(device)

                    with torch.cuda.amp.autocast():
                        output = model(image)
                        dice = 1 - dice_loss(output, mask, fov)
                        iou = iou_score(output, mask, fov)

                    dice_scores.append(dice.item())
                    ious.append(iou.item())

                    if idx < 5:
                        preds = torch.sigmoid(output) > 0.5
                        for i in range(image.size(0)):
                            input_img = TF.to_pil_image(image[i].cpu())
                            true_mask = TF.to_pil_image(mask[i].cpu())
                            pred_mask = TF.to_pil_image(preds[i].float().cpu())

                            input_img.save(f"{pred_dir}/img_{idx}_{i}_input.png")
                            true_mask.save(f"{pred_dir}/img_{idx}_{i}_mask.png")
                            pred_mask.save(f"{pred_dir}/img_{idx}_{i}_pred.png")

                        if not vis_logged:
                            log_images(image, mask, preds, epoch)
                            vis_logged = True

            avg_dice = sum(dice_scores) / len(dice_scores)
            avg_iou = sum(ious) / len(ious)

            writer.add_scalar("Dice/val", avg_dice, epoch)
            writer.add_scalar("IoU/val", avg_iou, epoch)

            print(f"📉 Loss={avg_train_loss:.4f} | 🎯 Dice={avg_dice:.4f} | 📈 IoU={avg_iou:.4f}")

            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), f"checkpoints/unet_best.pth")
                print("✅ Saved new best model.")

            torch.cuda.empty_cache()

    current_batch = batch_size
    current_size = base_image_size

    while True:
        try:
            attempt_training(current_batch, current_size)
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\n OOM: batch_size={current_batch}, image_size={current_size}")
                torch.cuda.empty_cache()
                if current_batch > 1:
                    current_batch = max(1, current_batch // 2)
                elif current_size > 256:
                    current_size = current_size // 2
                else:
                    raise RuntimeError("Cannot recover from OOM: batch_size=1 and image_size=256 failed")
                print(f"🔁 Retrying with batch_size={current_batch}, image_size={current_size}")
            else:
                raise e

    writer.close()