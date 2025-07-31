import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

def get_sam_transforms(size=1024):
    """
    Preprocessing pipeline za SAM.
    Samo resize, bez Normalize i bez ToTensorV2.
    Vraća uint8 numpy array.
    """
    return A.Compose([
        A.Resize(size, size)
    ])

class RetinaDatasetSAM2(Dataset):
    """
    Dataset specijalno prilagođen za SAM evaluaciju.
    - Radi resize na 1024x1024
    - NE radi normalizaciju niti ToTensor
    - Vraća slike kao uint8 numpy array
    - Čuva original_size da se predikcije mogu vratiti na pravi oblik
    """
    def __init__(self, image_dir, mask_dir, fov_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fov_dir = fov_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.fov_files = sorted(os.listdir(fov_dir)) if fov_dir is not None else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(image_path).convert("RGB"))  # uint8
        mask = np.array(Image.open(mask_path).convert("L"))      # 0/255
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

        if transform:
            transformed = transform(**data)
            image = transformed["image"]  # uint8 np.array
            mask = (transformed["mask"] > 0).astype(np.uint8)
            if fov is not None:
                fov = (transformed["fov"] > 0).astype(np.uint8)

        # Konvertuj image u float tensor i permutuј dimenzije za PyTorch: (H,W,3) -> (3,H,W)
        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        # Konvertuj mask i fov u float tensor, dimenzije (H,W) -> (1,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        if fov is not None:
            fov = torch.from_numpy(fov).unsqueeze(0).float()

        return {
            "image": image,                 
            "mask": mask,                   
            "fov": fov,                     
            "original_size": original_size, 
            "filename": self.image_files[idx]
        }