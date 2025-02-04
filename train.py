import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

def train_model(sam, train_loader, val_loader, device, epochs, learning_rate):
    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(sam.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        sam.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            batched_input = [{"image": img} for img in images]  # Convert images into the expected format
            outputs = sam(batched_input, multimask_output=False)

            loss = criterion(outputs, masks)

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
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = sam(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")