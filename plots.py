import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------
# PLOT 1: Bar chart po eksperimentima
# -------------------------
def plot_bar_chart(run_dirs, metric="best_iou", save_path="barchart.png"):
    """
    Plot bar chart for the chosen metric across different runs.
    """
    labels = []
    values = []

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            labels.append(os.path.basename(run_dir))
            values.append(summary.get(metric, 0.0))
        else:
            print(f"⚠️ Missing summary.json in {run_dir}")

    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y=values, palette="viridis")
    plt.title(f"Comparison of {metric} across runs")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved bar chart to {save_path}")

# -------------------------
# PLOT 2: Plot Loss / Dice / IoU from JSON logs
# -------------------------
def plot_metrics_from_json(run_dir, save_dir="plots"):
    """
    Plot metrics over epochs from the stored logs in metrics.json or tensorboard logs.
    """
    metrics_path = os.path.join(run_dir, "tensorboard")
    summary_path = os.path.join(run_dir, "summary.json")
    os.makedirs(save_dir, exist_ok=True)

    # Try to plot using saved metrics.json or logs
    if os.path.exists(summary_path):
        print(f"Reading summary from {summary_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print(json.dumps(summary, indent=2))

# -------------------------
# PLOT 3: Grid of images
# -------------------------
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

def plot_prediction_grid(pred_dir, save_path="pred_grid.png", max_images=5):
    """
    Plot a grid of images: input, mask, prediction side by side.
    """
    images = []
    files = os.listdir(pred_dir)
    files = sorted([f for f in files if f.endswith("_input.png")])

    for f in files[:max_images]:
        prefix = f.replace("_input.png", "")
        img_path = os.path.join(pred_dir, f"{prefix}_input.png")
        mask_path = os.path.join(pred_dir, f"{prefix}_mask.png")
        pred_path = os.path.join(pred_dir, f"{prefix}_pred.png")

        img = TF.to_tensor(Image.open(img_path)).permute(1, 2, 0)
        mask = TF.to_tensor(Image.open(mask_path))[0]
        pred = TF.to_tensor(Image.open(pred_path))[0]

        # Stack horizontally
        combined = np.concatenate([
            img.numpy(),
            np.stack([mask]*3, axis=-1),
            np.stack([pred]*3, axis=-1)
        ], axis=1)
        images.append(combined)

    # Combine vertically
    if images:
        grid = np.concatenate(images, axis=0)
        plt.figure(figsize=(12, len(images)*3))
        plt.imshow(grid)
        plt.axis("off")
        plt.title("Predictions Grid (Input | Mask | Pred)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved prediction grid to {save_path}")
    else:
        print(f"No images found in {pred_dir}")