import os
from torch.utils.data import DataLoader
from datasets import RetinaDataset, get_transforms

def prepare_dataloaders(data_dir, batch_size):
    transform = get_transforms()

    train_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "train/images"),
        mask_dir=os.path.join(data_dir, "train/mask"),
        transform=transform
    )

    val_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "val/images"),
        mask_dir=os.path.join(data_dir, "val/mask"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader