import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

def get_sam_transforms(size=1024):
    """
    Minimal preprocessing za SAM:
    - Resize na 1024x1024
    - Bez normalizacije, bez ToTensorV2
    - Vraća uint8 numpy array
    """
    return A.Compose([
        A.Resize(size, size)
    ])

class RetinaDatasetSAM(Dataset):
    """
    Dataset za SAM zero-shot evaluaciju:
    - Resize 1024x1024
    - Slike u uint8 RGB
    - Maske i FOV u uint8 (0/1)
    - original_size za vraćanje predikcija nazad
    """
    def __init__(self, image_dir, mask_dir, fov_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fov_dir = fov_dir
        self.transform = transform or get_sam_transforms()

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.fov_files = sorted(os.listdir(fov_dir)) if fov_dir else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(image_path).convert("RGB"))  # (H,W,3), uint8
        mask = np.array(Image.open(mask_path).convert("L"))      # (H,W), 0/255
        original_size = image.shape[:2]  # (H, W)

        fov = None
        if self.fov_dir:
            fov_path = os.path.join(self.fov_dir, self.fov_files[idx])
            fov = np.array(Image.open(fov_path).convert("L"))

            data = {"image": image, "mask": mask, "fov": fov}
            transform = A.Compose(
                self.transform.transforms,
                additional_targets={"fov": "mask"}
            )
        else:
            data = {"image": image, "mask": mask}
            transform = self.transform

        transformed = transform(**data)
        image = transformed["image"]
        mask = (transformed["mask"] > 0).astype(np.uint8)
        if fov is not None:
            fov = (transformed["fov"] > 0).astype(np.uint8)

        return {
            "image": image,                 # uint8 (H,W,3)
            "mask": mask,                   # uint8 (H,W)
            "fov": fov,                     # uint8 (H,W) ili None
            "original_size": original_size, # za resize
            "filename": self.image_files[idx]
        }