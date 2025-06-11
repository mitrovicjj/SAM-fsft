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
    

    def ensure_4d(t, like):
  
        if t.dim() == 2:             # H, W → 1, 1, H, W
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:           # (B or C), H, W  →       add missing dim
            if t.size(0) == like.size(-2):   # heuristic: treat leading dim as C
                t = t.unsqueeze(0)           #   add batch
            else:
                t = t.unsqueeze(1)           #   add channel
        # At this point t is [B, 1, H, W] or [B, C, H, W]
        if t.size(1) == 1:          # replicate channel to 3
            t = t.repeat(1, 3, 1, 1)
        return t


    def debug_log_images(images, masks, preds, epoch, tag="Validation"):
        print(f"Type masks before log_images: {type(masks)}, shape: {getattr(masks, 'shape', None)}")
        print(f"Type preds before log_images: {type(preds)}, shape: {getattr(preds, 'shape', None)}")
        return log_images(images, masks, preds, epoch, tag)

    def log_images(images, masks, preds, epoch, tag="Validation"):
          """
          Robust logger that works for any combination of 2‑D, 3‑D, or 4‑D tensors.
          All three tensors come out as [B, 3, H, W] on CPU.
          """

          def _prep(t, name):
              print(f"{name} IN  shape={tuple(t.shape)}  dim={t.dim()}")

              t = t.detach().cpu()       # (no grad, move-to‑CPU)
              if t.dim() == 2:           # H, W → 1,1,H,W
                  t = t.unsqueeze(0).unsqueeze(0)
              elif t.dim() == 3:         # could be C,H,W  or B,H,W
                  if t.size(0) in (1, 3):        # looks like C,H,W
                      t = t.unsqueeze(0)         # add batch
                  else:                          # looks like B,H,W
                      t = t.unsqueeze(1)         # add channel
              elif t.dim() != 4:
                  raise ValueError(f"Unexpected ndim {t.dim()} in {name}")

              # ensure 3 channels so cat() along batch works later
              if t.size(1) == 1:
                  t = t.repeat(1, 3, 1, 1)

              print(f"{name} OUT shape={tuple(t.shape)}  dim={t.dim()}")
              return t.float()           # TensorBoard dislikes bool

          img4   = _prep(images, "images")
          mask4  = _prep(masks,  "masks")
          pred4  = _prep(preds,   "preds")

          grid = make_grid(torch.cat([img4, mask4, pred4], dim=0),
                          nrow=img4.size(0))
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
                            preds = torch.sigmoid(output) > 0.5  # fresh preds
                            debug_log_images(image, mask, preds, epoch)
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