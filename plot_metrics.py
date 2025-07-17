import json
import os
import matplotlib.pyplot as plt

FOLDERS = {
    "FOV + AUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_fov_aug",
    "FOV + NoAUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_fov_noaug",
    "NoFOV + AUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_nofov_aug",
    "NoFOV + NoAUG": "C:\\Users\\Korisnik\\py\\SAM-fsft\\eval_seg_nofov_noaug"
}

metrics_to_plot = ["dice_mean", "iou_mean", "precision_mean", "recall_mean"]

def load_avg_metrics(folder):
    with open(os.path.join(folder, "metrics.json")) as f:
        metrics = json.load(f)
    return {k: metrics[k] for k in metrics_to_plot}

data = {label: load_avg_metrics(path) for label, path in FOLDERS.items()}
x = range(len(FOLDERS))
width = 0.2

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics_to_plot):
    plt.bar(
        [pos + i*width for pos in x],
        [data[label][metric] for label in FOLDERS],
        width=width,
        label=metric
    )

plt.xticks([pos + width*1.5 for pos in x], list(FOLDERS.keys()))
plt.ylabel("Metric Value")
plt.title("Average Metrics Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("barplot_average_metrics.png", dpi=300)
plt.show()