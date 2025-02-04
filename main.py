import argparse
import torch
from utils import prepare_dataloaders
from model import load_model
from train import train_model
from evaluate import visualize_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM on Retina Images")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to SAM checkpoint file")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader, val_loader = prepare_dataloaders(args.data_dir, args.batch_size)

    # Load SAM model
    sam = load_model(args.checkpoint, device)

    # Train the model
    train_model(sam, train_loader, val_loader, device, args.epochs, args.learning_rate)

    # Visualize results
    visualize_results(sam, val_loader, device)