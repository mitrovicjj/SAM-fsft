import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(size=256, is_train=True):

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if is_train:
        return A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.75),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.2, alpha=1, sigma=50, alpha_affine=50),
            A.GridDistortion(p=0.2),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])

class RetinaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, fov_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fov_dir = fov_dir
        self.transform = transform

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.fov_files = sorted(os.listdir(fov_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        fov = None
        original_size = image.shape[:2]

        if self.fov_dir:
            fov_path = os.path.join(self.fov_dir, self.fov_files[idx])
            fov = np.array(Image.open(fov_path).convert("L"))

            # Compose transform for fov as additional mask
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
            image = transformed["image"]
            # Convert mask and fov to binary float tensors
            mask = (transformed["mask"] > 0).float()
            if fov is not None:
                fov = (transformed["fov"] > 0).float()
        else:
            image = ToTensorV2()(image=image)["image"]
            mask = (ToTensorV2()(image=mask)["image"] > 0).float()
            if fov is not None:
                fov = (ToTensorV2()(image=fov)["image"] > 0).float()

        return {
            "image": image,
            "mask": mask,
            "fov": fov,
            "original_size": original_size
        }