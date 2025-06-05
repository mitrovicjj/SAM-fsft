import os
import torch
from tqdm import tqdm
from datasets import RetinaDataset, get_transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

save_dir = 'data/processed/training'
os.makedirs(save_dir, exist_ok=True)

def get_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

dataset = RetinaDataset(
    image_dir='data/raw/training/images',
    mask_dir='data/raw/training/masks',
    fov_dir='data/raw/training/fov',
    transform=get_transforms()
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, batch in enumerate(tqdm(loader)):
    img = batch['image'].squeeze(0)    # [3, 1024, 1024]
    mask = batch['mask'].squeeze(0)    # [1, 1024, 1024] → [1024, 1024]
    fov = batch['fov'].squeeze(0)

    sample = {
        'image': img,
        'mask': mask,
        'fov': fov
    }

    torch.save(sample, os.path.join(save_dir, f'sample_{i:03d}.pt'))