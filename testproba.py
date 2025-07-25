import re
import pandas as pd

log_text = """
🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:23<00:00,  1.25it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.39it/s]
[I 2025-07-17 17:59:28,895] Trial 14 finished with value: 0.5224039435386658 and parameters: {'batch_size': 1, 'lr': 0.004797176509701105, 'weight_decay': 0.0005526662837063506, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.5047 | Val Loss=0.4973 | 🎯 Dice=0.3149 | 📈 IoU=0.4814 | 🔍 Prec=0.6539 | 🧠 Rec=0.6468
⚠️ No improvement. wait_counter=12/12
"""

epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
metrics_pattern = re.compile(
    r"Train Loss=([\d\.]+) \| Val Loss=([\d\.]+) \| 🎯 Dice=([\d\.]+) \| 📈 IoU=([\d\.]+) \| 🔍 Prec=([\d\.]+) \| 🧠 Rec=([\d\.]+)"
)

epochs = []
train_losses = []
val_losses = []
dices = []
ious = []
precisions = []
recalls = []

lines = log_text.splitlines()

for i, line in enumerate(lines):
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        for j in range(i+1, min(i+6, len(lines))):
            metric_match = metrics_pattern.search(lines[j])
            if metric_match:
                epochs.append(epoch_num)
                train_losses.append(float(metric_match.group(1)))
                val_losses.append(float(metric_match.group(2)))
                dices.append(float(metric_match.group(3)))
                ious.append(float(metric_match.group(4)))
                precisions.append(float(metric_match.group(5)))
                recalls.append(float(metric_match.group(6)))
                break

df = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "dice": dices,
    "iou": ious,
    "precision": precisions,
    "recall": recalls
})

print(df)
df.to_csv("epoch_metrics_unet_new.csv", index=False)