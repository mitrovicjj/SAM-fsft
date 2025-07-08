import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
from unetmodel import UNet
from datasets import get_transforms, RetinaDataset
from utils import prepare_dataloaders
from metrics import dice_loss, iou_score
import torchvision.transforms.functional as TF
from PIL import Image


def train_model(data_dir, epochs, batch_size, lr, device='cuda', log_dir=None):
    """Top-level training entry point.

    Args:
        data_dir: Root folder with train and val subfolders.
        epochs: Number of epochs for each resolution stage.
        batch_size: Initial batch size (auto‑downscales on OOM).
        lr: Adam learning rate.
        device: "cuda" or "cpu".
        log_dir: Root directory for logging runs (auto-created if None).
    """

    # === Set‑up ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unet_bs{batch_size}_lr{lr}_ep{epochs}_{timestamp}"
    run_dir = log_dir or os.path.join("runs", run_name)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    predictions_dir = os.path.join(run_dir, "predictions")

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save training config
    config = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "run_dir": run_dir
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    scaler = torch.amp.GradScaler(device)
    best_iou = -1.0

    base_image_size = 256
    accumulation_steps = 4

    # ------------------------------------------------------------------
    # Helper: robust image logger
    # ------------------------------------------------------------------
    def log_images(images, masks, preds, epoch, tag: str = "Validation"):
        def _prep(t: torch.Tensor, name: str) -> torch.Tensor:
            t = t.detach().cpu()
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 3:
                if t.size(0) in (1, 3):
                    t = t.unsqueeze(0)
                else:
                    t = t.unsqueeze(1)
            elif t.dim() != 4:
                raise ValueError(f"Unexpected ndim {t.dim()} in {name}")
            if t.size(1) == 1:
                t = t.repeat(1, 3, 1, 1)
            return t.float()

        img4 = _prep(images, "images")
        mask4 = _prep(masks, "masks")
        pred4 = _prep(preds, "preds")
        grid = make_grid(torch.cat([img4, mask4, pred4], dim=0), nrow=img4.size(0))
        writer.add_image(f"{tag}/image-mask-pred", grid, epoch)

    # ------------------------------------------------------------------
    # Inner training routine (allows automatic resolution / BS fallback)
    # ------------------------------------------------------------------
    def attempt_training(current_bs: int, image_size: int):
        nonlocal best_iou

        print(f"\n➤ Starting training: batch_size={current_bs}, image_size={image_size}")
        transform = get_transforms(image_size)
        train_loader, val_loader = prepare_dataloaders(data_dir, current_bs, transform)

        model = UNet(n_channels=3, n_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
            model.train()
            cum_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                fov = fov.to(device) if fov is not None else None

                with torch.amp.autocast(device_type=device):
                    out = model(img)
                    loss = dice_loss(out, mask, fov) / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                cum_loss += loss.item() * accumulation_steps

            avg_train_loss = cum_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation
            model.eval()
            dice_scores, ious = [], []
            vis_logged = False
            epoch_dir = os.path.join(predictions_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                    img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                    fov = fov.to(device) if fov is not None else None

                    with torch.amp.autocast(device_type=device):
                        out = model(img)
                        dice = 1 - dice_loss(out, mask, fov)
                        iou = iou_score(out, mask, fov)

                    dice_scores.append(dice.item())
                    ious.append(iou.item())

                    if idx < 5:
                        preds = torch.sigmoid(out) > 0.5
                        for i in range(img.size(0)):
                            TF.to_pil_image(img[i].cpu()).save(f"{epoch_dir}/img_{idx}_{i}_input.png")
                            TF.to_pil_image(mask[i].cpu()).save(f"{epoch_dir}/img_{idx}_{i}_mask.png")
                            TF.to_pil_image(preds[i].float().cpu()).save(f"{epoch_dir}/img_{idx}_{i}_pred.png")

                        if not vis_logged:
                            log_images(img, mask, preds, epoch)
                            vis_logged = True

            avg_dice = sum(dice_scores) / len(dice_scores)
            avg_iou = sum(ious) / len(ious)
            writer.add_scalar("Dice/val", avg_dice, epoch)
            writer.add_scalar("IoU/val", avg_iou, epoch)

            print(f"📉 Loss={avg_train_loss:.4f} | 🎯 Dice={avg_dice:.4f} | 📈 IoU={avg_iou:.4f}")

            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_best.pth"))
                print("✅ Saved new best model.")

            torch.cuda.empty_cache()

    cur_bs, cur_size = batch_size, base_image_size
    while True:
        try:
            attempt_training(cur_bs, cur_size)
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\n⚠️  OOM at bs={cur_bs}, img={cur_size}. Retrying...")
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                elif cur_size > 256:
                    cur_size //= 2
                else:
                    raise RuntimeError("Cannot recover from OOM: batch_size=1 and image_size=256 failed")
                print(f"🔁 New attempt: bs={cur_bs}, img={cur_size}")
            else:
                raise

    writer.close()