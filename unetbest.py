import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torch.utils.tensorboard import SummaryWriter
from unetmodel import UNet
from datasets import RetinaDataset, get_transforms
from metrics import dice_loss, iou_score
from tqdm import tqdm
from torchvision.utils import make_grid


def unnormalize(imgs, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs.clamp(0, 1)

def _prep_vis(t: torch.Tensor) -> torch.Tensor:
    """tensor for visualization: [B, 3, H, W]."""
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

def test_model(run_dir, data_dir, checkpoint_path=None,
               batch_size=None, image_size=None, device="cuda",
               save_predictions=True, output_dir=None, log_dir=None):

    # Load config.yaml if batch_size or image_size not passed explicitly
    if batch_size is None or image_size is None or (output_dir is None or log_dir is None):
        config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if batch_size is None:
                batch_size = config.get("batch_size", 1)
            if image_size is None:
                image_size = config.get("base_image_size", 256)
            if output_dir is None:
                output_dir = os.path.join(run_dir, "test_predictions")
            if log_dir is None:
                log_dir = os.path.join(run_dir, "tensorboard_test")
        else:
            print(f"Warning: config.yaml not found in {run_dir}, using defaults or provided args.")
            batch_size = batch_size or 1
            image_size = image_size or 256
            output_dir = output_dir or "output/testing_preds"
            log_dir = log_dir or None
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_dir, "checkpoints", "unet_best.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Testing model from checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}, Image size: {image_size}")
    print(f"Saving predictions to: {output_dir}")
    if log_dir:
        print(f"TensorBoard logs directory: {log_dir}")

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if log_dir else None

    transform = get_transforms(image_size)
    test_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "images"),
        mask_dir=os.path.join(data_dir, "masks"),
        fov_dir=os.path.join(data_dir, "fov"),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dice_scores, iou_scores = [], []
    vis_logged = False

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing UNet")):
            img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
            fov = fov.to(device) if fov is not None else None

            with torch.amp.autocast(device_type=device):
                out = model(img)
                dice = 1 - dice_loss(out, mask, fov)
                iou = iou_score(out, mask, fov)

            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

            preds = torch.sigmoid(out) > 0.5

            if save_predictions:
                img_unnorm = unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for i in range(img.size(0)):
                    TF.to_pil_image(img_unnorm[i].cpu()).save(f"{output_dir}/img_{idx}_{i}_input.png")
                    TF.to_pil_image(mask[i].cpu()).save(f"{output_dir}/img_{idx}_{i}_mask.png")
                    TF.to_pil_image(preds[i].float().cpu()).save(f"{output_dir}/img_{idx}_{i}_pred.png")

            # Log batch to TensorBoard
            if writer and not vis_logged:
                img_unnorm = unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img_vis = _prep_vis(img_unnorm)
                mask_vis = _prep_vis(mask)
                pred_vis = _prep_vis(preds)
                grid = make_grid(torch.cat([img_vis, mask_vis, pred_vis], dim=0), nrow=img.size(0))
                writer.add_image("Test/image-mask-pred", grid, 0)
                vis_logged = True


    mean_dice = sum(dice_scores) / len(dice_scores)
    mean_iou = sum(iou_scores) / len(iou_scores)

    if writer:
        writer.add_scalar("Test/Dice", mean_dice, 0)
        writer.add_scalar("Test/IoU", mean_iou, 0)
        writer.close()

    print(f"\nTest Dice Score: {mean_dice:.4f}")
    print(f"Test IoU Score : {mean_iou:.4f}")