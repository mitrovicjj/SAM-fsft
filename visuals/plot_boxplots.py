import os
import json
import matplotlib.pyplot as plt

FOLDERS = {
    "FOV + AUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_fov_aug",
    "FOV + NoAUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_fov_noaug",
    "NoFOV + AUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_nofov_aug",
    "NoFOV + NoAUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_nofov_noaug"
}

available_metrics = ["dice_per_image", "iou_per_image"]

def load_per_image_metrics(folder):
    with open(os.path.join(folder, "metrics.json")) as f:
        metrics = json.load(f)
    return {k: metrics[k] for k in available_metrics if k in metrics}

data = {label: load_per_image_metrics(path) for label, path in FOLDERS.items()}

for metric in available_metrics:
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [data[label][metric] for label in FOLDERS],
        labels=list(FOLDERS.keys()),
        showmeans=True
    )
    plt.title(f"{metric.replace('_per_image', '').capitalize()} per Image")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"boxplot_{metric}.png", dpi=300)
    plt.show()