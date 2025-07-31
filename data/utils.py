import os
from torch.utils.data import DataLoader
from datasets import RetinaDataset, get_transforms

def prepare_dataloaders(data_dir, batch_size):
    transform = get_transforms()

    train_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "train/images"),
        mask_dir=os.path.join(data_dir, "train/masks"),
        fov_dir=os.path.join(data_dir, "train/fov"),
        transform=transform
    )

    test_dataset = RetinaDataset(
        image_dir=os.path.join(data_dir, "test/images"),
        mask_dir=os.path.join(data_dir, "test/masks"),
        fov_dir=os.path.join(data_dir, "test/fov"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader