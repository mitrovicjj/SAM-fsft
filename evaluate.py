import os
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from dataset import get_transforms
from utils import prepare_dataloaders
from utils import prepare_test_dataloader
from segformermodel import get_segformer_model
from unetmodel import UNet
from metrics import dice_loss, iou_score, precision_score, recall_score

# --------------------------
# Helper: load model
# --------------------------
def load_model(model_type, backbone, checkpoint_path, device):
    if model_type == "segformer":
        model = get_segformer_model(backbone, num_labels=1)
    elif model_type == "unet":
        model = UNet(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# --------------------------
# Helper: single predict
# --------------------------
def predict(model, img, model_type):
    if model_type == "segformer":
        return model(img).logits
    elif model_type == "unet":
        return model(img)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# --------------------------
# Main Evaluation
# --------------------------
def evaluate(model, dataloader, device, model_type, output_dir):
    dice_list, iou_list, precision_list, recall_list = [], [], [], []
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(output_dir, "predictions")
    os.makedirs(img_dir, exist_ok=True)

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        img, mask, fov = batch["image"].to(device), batch["mask"].to(device), batch["fov"]
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if fov is not None and fov.dim() == 3:
            fov = fov.unsqueeze(1)
        fov = fov.to(device) if fov is not None else None

        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            out = predict(model, img, model_type)
            out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
            
            # TTA horizontal flip
            out_hflip = predict(model, img.flip(-1), model_type).flip(-1)
            out_hflip = F.interpolate(out_hflip, size=mask.shape[2:], mode='bilinear', align_corners=False)
            out = (out + out_hflip) / 2

            if fov is not None:
                out = out * fov
                mask = mask * fov

            dice = 1 - dice_loss(out, mask)
            iou = iou_score(out, mask)
            precision = precision_score(out, mask)
            recall = recall_score(out, mask)

        dice_list.append(dice.item())
        iou_list.append(iou.item())
        precision_list.append(precision.item())
        recall_list.append(recall.item())

        pred_bin = (torch.sigmoid(out) > 0.5).float()
        for i in range(img.size(0)):
            TF.to_pil_image(img[i].cpu()).save(os.path.join(img_dir, f"{idx}_{i}_input.png"))
            TF.to_pil_image(mask[i].cpu()).save(os.path.join(img_dir, f"{idx}_{i}_mask.png"))
            TF.to_pil_image(pred_bin[i].cpu()).save(os.path.join(img_dir, f"{idx}_{i}_pred.png"))

    results = dict(
        dice_mean = sum(dice_list) / len(dice_list),
        iou_mean = sum(iou_list) / len(iou_list),
        precision_mean = sum(precision_list) / len(precision_list),
        recall_mean = sum(recall_list) / len(recall_list),
        dice_per_image = dice_list,
        iou_per_image = iou_list
    )

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation Done!")
    print(json.dumps(results, indent=2))

    # Plot histograms
    plt.figure(figsize=(6,4))
    plt.hist(dice_list, bins=20, alpha=0.7, color='skyblue')
    plt.title("Dice Coefficient Distribution")
    plt.xlabel("Dice")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "dice_hist.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(iou_list, bins=20, alpha=0.7, color='salmon')
    plt.title("IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "iou_hist.png"))
    plt.close()

# --------------------------
# CLI Entry
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["unet", "segformer"], required=True)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_type, args.backbone, args.checkpoint, device)

    if args.model_type == "segformer":
        size = 512
    else:
        size = 256

    val_transform = get_transforms(size=size, is_train=False)

    val_loader = prepare_test_dataloader(
        data_dir,
        batch_size,
        get_transforms(size, is_train=False)
    )

    evaluate(model, val_loader, device, args.model_type, args.output_dir)


if __name__ == "__main__":
    main()