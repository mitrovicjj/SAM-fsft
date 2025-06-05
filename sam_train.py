import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

def train_model(sam, train_loader, val_loader, device, epochs, learning_rate):
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(sam.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        sam.train()
        train_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            original_sizes = batch["original_size"]  # Keep this as metadata

            # Create batched_input with original_size included
            batched_input = [
                {"image": img, "original_size": orig_size} 
                for img, orig_size in zip(images, original_sizes)
            ]

            outputs = sam(batched_input, multimask_output=False)

            # Extract model predictions
            predicted_masks = torch.stack([out["masks"] for out in outputs]).squeeze(1)  # Shape: (B, H, W)

            loss = criterion(predicted_masks, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")
        validate_model(sam, val_loader, device, criterion)

def validate_model(sam, val_loader, device, criterion):
    sam.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            original_sizes = batch["original_size"]

            batched_input = [
                {"image": img, "original_size": orig_size} 
                for img, orig_size in zip(images, original_sizes)
            ]

            outputs = sam(batched_input, multimask_output=False)
            predicted_masks = torch.stack([out["masks"] for out in outputs]).squeeze(1)

            loss = criterion(predicted_masks, masks)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")