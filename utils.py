import os
from torch.utils.data import DataLoader
from datasets import RetinaDataset, get_transforms

def prepare_dataloaders(data_dir, batch_size, train_transform, val_transform):
    train_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "train/images"),
        mask_dir=os.path.join(data_dir, "train/masks"),
        fov_dir=os.path.join(data_dir, "train/fov"),
        transform=train_transform
    )

    val_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "test/images"),
        mask_dir=os.path.join(data_dir, "test/masks"),
        fov_dir=os.path.join(data_dir, "test/fov"),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader