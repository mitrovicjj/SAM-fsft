import os
import cv2
import matplotlib.pyplot as plt

FOLDERS = {
    "FOV + AUG": "eval_seg_fov_aug/predictions",
    "FOV + NoAUG": "eval_seg_fov_noaug/predictions",
    "NoFOV + AUG": "eval_seg_nofov_aug/predictions",
    "NoFOV + NoAUG": "eval_seg_nofov_noaug/predictions"
}

example_image_name = "4_0_pred.png"

plt.figure(figsize=(12, 6))
for i, (label, pred_path) in enumerate(FOLDERS.items()):
    img = cv2.imread(os.path.join(pred_path, example_image_name), cv2.IMREAD_GRAYSCALE)
    full_path = os.path.join(pred_path, example_image_name)
    print(f"Loading image: {full_path}")
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {full_path}"
    plt.subplot(2, 2, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis("off")

plt.suptitle(f"Prediction Comparison: {example_image_name}")
plt.tight_layout()
plt.savefig(f"comparison_{example_image_name}.png", dpi=300)
plt.show()