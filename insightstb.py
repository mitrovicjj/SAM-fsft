import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_line(ea, tag, save_path):
    events = ea.Scalars(tag)
    if not events:
        print(f"Tag {tag} not found.")
        return
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.figure()
    plt.plot(steps, values, marker="o")
    plt.title(f"{tag}")
    plt.xlabel("Epoch")
    plt.ylabel(tag)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="insight_plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    ea = event_accumulator.EventAccumulator(args.log_dir)
    ea.Reload()

    tags = ["Loss/train", "Dice/val", "IoU/val"]
    for tag in tags:
        plot_line(
            ea,
            tag,
            os.path.join(args.output, f"{tag.replace('/','_')}.png")
        )

    # Scatter IoU vs. Dice
    dice_events = ea.Scalars("Dice/val")
    iou_events = ea.Scalars("IoU/val")
    if dice_events and iou_events:
        dice_vals = [e.value for e in dice_events]
        iou_vals = [e.value for e in iou_events]
        plt.figure()
        plt.scatter(dice_vals, iou_vals, color="teal")
        plt.xlabel("Dice")
        plt.ylabel("IoU")
        plt.title("IoU vs Dice scatter")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "scatter_iou_vs_dice.png"))
        plt.close()
        print("Saved scatter plot IoU vs Dice.")

if __name__ == "__main__":
    main()