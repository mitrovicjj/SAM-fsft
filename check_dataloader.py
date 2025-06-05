from utils import prepare_dataloaders

train_loader, test_loader = prepare_dataloaders("data/raw", batch_size=4)

for batch in train_loader:
    print("Image shape:", batch["image"].shape)  # Expect [B, 3, 1024, 1024]
    print("Mask shape:", batch["mask"].shape)    # Expect [B, 1, 1024, 1024] or [B, 1024, 1024]
    print("FOV shape:", batch["fov"].shape if batch["fov"] is not None else None)
    print("Pixel range:", batch["image"].min().item(), batch["image"].max().item())
    break