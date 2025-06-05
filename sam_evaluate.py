import matplotlib.pyplot as plt
import torch

def visualize_results(sam, loader, device):
    sam.eval()
    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            image, mask = image.to(device), mask.to(device)
            predicted_masks = sam(image)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Input Image")

            plt.subplot(1, 3, 2)
            plt.imshow(mask[0].cpu().numpy(), cmap="gray")
            plt.title("Ground Truth")

            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[0].cpu().numpy(), cmap="gray")
            plt.title("Predicted Mask")

            plt.show()
            if idx == 0: break