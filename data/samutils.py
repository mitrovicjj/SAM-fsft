import os
from torch.utils.data import DataLoader
from data.samdataset2 import RetinaDatasetSAM, get_sam_transforms

def prepare_dataloaders(data_dir, batch_size, train_transform, val_transform):
    train_dataset = RetinaDatasetSAM(
        image_dir=os.path.join(data_dir, "train/images"),
        mask_dir=os.path.join(data_dir, "train/masks"),
        fov_dir=os.path.join(data_dir, "train/fov"),
        transform=train_transform
    )

    val_dataset = RetinaDatasetSAM(
        image_dir=os.path.join(data_dir, "test/images"),
        mask_dir=os.path.join(data_dir, "test/masks"),
        fov_dir=os.path.join(data_dir, "test/fov"),
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def prepare_test_dataloader(data_dir, batch_size, transform):
    dataset = RetinaDatasetSAM(
        image_dir=os.path.join(data_dir, "images"),
        mask_dir=os.path.join(data_dir, "masks"),
        fov_dir=os.path.join(data_dir, "fov"),
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader