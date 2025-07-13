import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import yaml
import json
import torchvision.transforms.functional as TF
from segformermodel import get_segformer_model
from unetmodel import UNet
from dataset import get_transforms
from utils import prepare_dataloaders
from metrics import dice_loss, iou_score, precision_score, recall_score

# -----------------------------------------------------
# Helper: unnormalize images for visualization
# -----------------------------------------------------
def unnormalize(imgs, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs.clamp(0, 1)

# -----------------------------------------------------
# Unified predict function
# -----------------------------------------------------
def predict(model, img, model_type):
    if model_type == "segformer":
        return model(img).logits
    elif model_type == "unet":
        return model(img)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# -----------------------------------------------------
# Get model by name
# -----------------------------------------------------
def get_model(name, backbone, num_labels):
    if name == "segformer":
        return get_segformer_model(backbone, num_labels)
    elif name == "unet":
        return UNet(n_channels=3, n_classes=num_labels)
    else:
        raise ValueError(f"Unknown model name: {name}")

# -----------------------------------------------------
# Main training function
# -----------------------------------------------------
def train_model(
    data_dir,
    model_type,
    backbone,
    epochs,
    batch_size,
    lr,
    device="cuda",
    log_dir=None,
    accumulation_steps=4,
    patience=10,
    seed=42
):
    # ---------------------
    # SEED for reproducibility
    # ---------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------------
    # Run directories
    # ---------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_bs{batch_size}_lr{lr}_ep{epochs}_{timestamp}"
    run_dir = log_dir or os.path.join("runs", run_name)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    predictions_dir = os.path.join(run_dir, "predictions")

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Save parameters
    param_dict = dict(
        model_type=model_type,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        accumulation_steps=accumulation_steps,
        patience=patience,
        seed=seed
    )
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(param_dict, f)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    scaler = torch.amp.GradScaler(device)
    best_iou = -1.0
    best_epoch = -1
    wait_counter = 0

    base_image_size = 512 if model_type == "segformer" else 256

    # -------------------------------------------------------
    # Logging helper
    # -------------------------------------------------------
    def log_images(images, masks, preds, epoch, tag="Validation"):
        images = unnormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        def _prep(t: torch.Tensor, name: str):
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

        grid = make_grid(
            torch.cat([
                _prep(images, "images"),
                _prep(masks.float(), "masks"),
                _prep(preds.float(), "preds")
            ], dim=0),
            nrow=images.size(0)
        )
        writer.add_image(f"{tag}/image-mask-pred", grid, epoch)

    # -------------------------------------------------------
    # Inner training function
    # -------------------------------------------------------
    def attempt_training(current_bs, current_size):
        nonlocal best_iou, wait_counter, best_epoch

        print(f"\n➤ Starting training: model={model_type}, backbone={backbone}, bs={current_bs}, img={current_size}")

        train_transform = get_transforms(size=current_size, is_train=True)
        val_transform = get_transforms(size=current_size, is_train=False)

        train_loader, val_loader = prepare_dataloaders(data_dir, current_bs, train_transform, val_transform)

        model = get_model(model_type, backbone, num_labels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Log hyperparameters
        writer.add_hparams(
            {
                "model_type": model_type,
                "backbone": backbone,
                "lr": lr,
                "bs": current_bs,
                "epochs": epochs
            },
            {}
        )

        for epoch in range(epochs):
            print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
            model.train()
            cum_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if fov is not None and fov.dim() == 3:
                    fov = fov.unsqueeze(1)
                fov = fov.to(device) if fov is not None else None

                with torch.amp.autocast(device_type=device):
                    out = predict(model, img, model_type)
                    out = torch.nn.functional.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)

                    # apply fov mask if exists
                    if fov is not None:
                        out = out * fov
                        mask = mask * fov

                    bce = nn.functional.binary_cross_entropy_with_logits(out, mask)
                    dice = dice_loss(out, mask)
                    loss = (0.6 * bce + 0.4 * dice) / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()

                    if model_type == "unet":
                        for name, param in model.named_parameters():
                            writer.add_histogram(f"Weights/{name}", param, epoch)
                            if param.grad is not None:
                                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

                    optimizer.zero_grad()

                cum_loss += loss.item() * accumulation_steps

            avg_train_loss = cum_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation
            model.eval()
            dice_scores, ious, precisions, recalls = [], [], [], []
            vis_logged = False
            epoch_dir = os.path.join(predictions_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
                    img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(1)
                    if fov is not None and fov.dim() == 3:
                        fov = fov.unsqueeze(1)
                    fov = fov.to(device) if fov is not None else None

                    with torch.amp.autocast(device_type=device):
                        out = predict(model, img, model_type)
                        out = torch.nn.functional.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)

                        out_hflip = predict(model, img.flip(-1), model_type).flip(-1)
                        out_hflip = torch.nn.functional.interpolate(out_hflip, size=mask.shape[2:], mode='bilinear', align_corners=False)
                        out = (out + out_hflip) / 2

                        # apply fov mask if exists
                        if fov is not None:
                            out = out * fov
                            mask = mask * fov

                        dice = 1 - dice_loss(out, mask)
                        iou = iou_score(out, mask)
                        precision = precision_score(out, mask)
                        recall = recall_score(out, mask)

                    dice_scores.append(dice.item())
                    ious.append(iou.item())
                    precisions.append(precision.item())
                    recalls.append(recall.item())

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
            avg_prec = sum(precisions) / len(precisions)
            avg_rec = sum(recalls) / len(recalls)

            writer.add_scalar("Dice/val", avg_dice, epoch)
            writer.add_scalar("IoU/val", avg_iou, epoch)
            writer.add_scalar("Precision/val", avg_prec, epoch)
            writer.add_scalar("Recall/val", avg_rec, epoch)

            print(f"📉 Loss={avg_train_loss:.4f} | 🎯 Dice={avg_dice:.4f} | 📈 IoU={avg_iou:.4f} | 🔍 Prec={avg_prec:.4f} | 🧠 Rec={avg_rec:.4f}")

            # Early stopping logic
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_epoch = epoch
                wait_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': best_epoch,
                    'iou': best_iou,
                    'params': param_dict,
                }, os.path.join(checkpoint_dir, f"{model_type}_best.pth"))
                print("✅ Saved new best model.")
            else:
                wait_counter += 1
                print(f"⚠️ No improvement. wait_counter={wait_counter}/{patience}")

            if wait_counter >= patience:
                print("⏹ Early stopping triggered!")
                break

            torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Fallback logic for OOM
    # -------------------------------------------------------
    cur_bs, cur_size = batch_size, base_image_size
    while True:
        try:
            attempt_training(cur_bs, cur_size)
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\n⚠️ OOM at bs={cur_bs}, img={cur_size}. Retrying...")
                torch.cuda.empty_cache()
                if cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                elif model_type == "unet" and cur_size > 256:
                    cur_size //= 2
                else:
                    raise RuntimeError("Cannot recover from OOM.")
                print(f"🔁 New attempt: bs={cur_bs}, img={cur_size}")
            else:
                raise

    writer.close()

    # summary.json
    summary = {
        "best_epoch": best_epoch,
        "best_iou": best_iou,
        "run_dir": run_dir
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary