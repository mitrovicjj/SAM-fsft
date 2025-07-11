import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
from segformermodel import get_segformer_model
from dataset import get_transforms, RetinaDataset
from utils import prepare_dataloaders
from metrics import dice_loss, iou_score, precision_score, recall_score
import torchvision.transforms.functional as TF


def unnormalize(imgs, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs.clamp(0, 1)


def train_segformer(cfg_path: str, device='cuda'):
    # === Config ===
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    image_size = cfg["image_size"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    data_dir = cfg["data_dir"]
    pretrained_model = cfg["pretrained_model"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"segformer_bs{batch_size}_lr{lr}_ep{epochs}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    predictions_dir = os.path.join(run_dir, "predictions")

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    scaler = torch.amp.GradScaler(device)
    best_iou = -1.0
    accumulation_steps = 4

    def log_images(images, masks, preds, epoch, tag="Validation"):
        images = unnormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def _prep(t):
            t = t.detach().cpu()
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 3:
                if t.size(0) in (1, 3):
                    t = t.unsqueeze(0)
                else:
                    t = t.unsqueeze(1)
            if t.size(1) == 1:
                t = t.repeat(1, 3, 1, 1)
            return t.float()

        grid = make_grid(torch.cat([
            _prep(images),
            _prep(masks),
            _prep(preds)
        ], dim=0), nrow=images.size(0))
        writer.add_image(f"{tag}/image-mask-pred", grid, epoch)

    def attempt_training(current_bs: int):
        nonlocal best_iou

        print(f"\n➤ Starting training: batch_size={current_bs}, image_size={image_size}")

        train_transform = get_transforms(size=image_size, is_train=True)
        val_transform = get_transforms(size=image_size, is_train=False)

        train_loader, val_loader = prepare_dataloaders(data_dir, current_bs, train_transform, val_transform)

        model = get_segformer_model(pretrained_model, num_labels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
                    out = model(img).logits
                    out = torch.nn.functional.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)

                    bce = nn.functional.binary_cross_entropy_with_logits(out * fov, mask * fov) if fov is not None \
                        else nn.functional.binary_cross_entropy_with_logits(out, mask)
                    dice = dice_loss(out, mask, fov)
                    loss = (0.6 * bce + 0.4 * dice) / acc