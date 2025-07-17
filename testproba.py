import re
import pandas as pd

log_text = """
🔁 Epoch 1/25
Training: 100%|██████████| 15/15 [00:55<00:00,  3.70s/it]
Validating: 100%|██████████| 3/3 [00:06<00:00,  2.24s/it]
📉 Train Loss=0.7302 | Val Loss=0.6946 | 🎯 Dice=0.1398 | 📈 IoU=0.0182 | 🔍 Prec=0.2879 | 🧠 Rec=0.0189
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.52it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.14it/s]
📉 Train Loss=0.6925 | Val Loss=0.6771 | 🎯 Dice=0.1445 | 📈 IoU=0.0755 | 🔍 Prec=0.5124 | 🧠 Rec=0.0812
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 15/15 [00:08<00:00,  1.74it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.59it/s]
📉 Train Loss=0.6788 | Val Loss=0.6654 | 🎯 Dice=0.1542 | 📈 IoU=0.1795 | 🔍 Prec=0.5069 | 🧠 Rec=0.2187
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 15/15 [00:09<00:00,  1.54it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.99it/s]
📉 Train Loss=0.6644 | Val Loss=0.6549 | 🎯 Dice=0.1692 | 📈 IoU=0.2552 | 🔍 Prec=0.4735 | 🧠 Rec=0.3590
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.49it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.06it/s]
📉 Train Loss=0.6570 | Val Loss=0.6430 | 🎯 Dice=0.1832 | 📈 IoU=0.3152 | 🔍 Prec=0.5199 | 🧠 Rec=0.4460
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.48it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.78it/s]
📉 Train Loss=0.6448 | Val Loss=0.6329 | 🎯 Dice=0.2007 | 📈 IoU=0.3520 | 🔍 Prec=0.5214 | 🧠 Rec=0.5226
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 15/15 [00:08<00:00,  1.70it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.52it/s]
📉 Train Loss=0.6363 | Val Loss=0.6256 | 🎯 Dice=0.2066 | 📈 IoU=0.3754 | 🔍 Prec=0.5570 | 🧠 Rec=0.5371
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.56it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.90it/s]
📉 Train Loss=0.6294 | Val Loss=0.6200 | 🎯 Dice=0.2096 | 📈 IoU=0.3887 | 🔍 Prec=0.5758 | 🧠 Rec=0.5464
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.50it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.68it/s]
📉 Train Loss=0.6229 | Val Loss=0.6116 | 🎯 Dice=0.2256 | 📈 IoU=0.4129 | 🔍 Prec=0.5449 | 🧠 Rec=0.6330
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.47it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.98it/s]
📉 Train Loss=0.6175 | Val Loss=0.6062 | 🎯 Dice=0.2274 | 📈 IoU=0.4285 | 🔍 Prec=0.5757 | 🧠 Rec=0.6286
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 15/15 [00:08<00:00,  1.74it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.50it/s]
📉 Train Loss=0.6155 | Val Loss=0.6027 | 🎯 Dice=0.2331 | 📈 IoU=0.4335 | 🔍 Prec=0.5687 | 🧠 Rec=0.6482
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.56it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.77it/s]
📉 Train Loss=0.6095 | Val Loss=0.6019 | 🎯 Dice=0.2325 | 📈 IoU=0.4291 | 🔍 Prec=0.5751 | 🧠 Rec=0.6306
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 13/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.50it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.79it/s]
📉 Train Loss=0.6058 | Val Loss=0.5960 | 🎯 Dice=0.2394 | 📈 IoU=0.4457 | 🔍 Prec=0.5782 | 🧠 Rec=0.6624
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.50it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.58it/s]
📉 Train Loss=0.6029 | Val Loss=0.5932 | 🎯 Dice=0.2426 | 📈 IoU=0.4512 | 🔍 Prec=0.5840 | 🧠 Rec=0.6672
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 15/15 [00:08<00:00,  1.68it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.19it/s]
📉 Train Loss=0.5953 | Val Loss=0.5892 | 🎯 Dice=0.2480 | 📈 IoU=0.4591 | 🔍 Prec=0.5846 | 🧠 Rec=0.6836
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.56it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.68it/s]
📉 Train Loss=0.5974 | Val Loss=0.5883 | 🎯 Dice=0.2452 | 📈 IoU=0.4630 | 🔍 Prec=0.5988 | 🧠 Rec=0.6731
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.44it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.73it/s]
📉 Train Loss=0.5909 | Val Loss=0.5890 | 🎯 Dice=0.2465 | 📈 IoU=0.4494 | 🔍 Prec=0.5825 | 🧠 Rec=0.6648
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 18/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.49it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.94it/s]
📉 Train Loss=0.5926 | Val Loss=0.5823 | 🎯 Dice=0.2527 | 📈 IoU=0.4731 | 🔍 Prec=0.5983 | 🧠 Rec=0.6949
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.60it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.31it/s]
📉 Train Loss=0.5891 | Val Loss=0.5831 | 🎯 Dice=0.2485 | 📈 IoU=0.4640 | 🔍 Prec=0.6035 | 🧠 Rec=0.6697
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 20/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.60it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.93it/s]
📉 Train Loss=0.5909 | Val Loss=0.5804 | 🎯 Dice=0.2557 | 📈 IoU=0.4684 | 🔍 Prec=0.5900 | 🧠 Rec=0.6962
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 21/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.44it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.66it/s]
📉 Train Loss=0.5839 | Val Loss=0.5771 | 🎯 Dice=0.2530 | 📈 IoU=0.4815 | 🔍 Prec=0.6272 | 🧠 Rec=0.6762
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.45it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.84it/s]
📉 Train Loss=0.5810 | Val Loss=0.5747 | 🎯 Dice=0.2625 | 📈 IoU=0.4773 | 🔍 Prec=0.5919 | 🧠 Rec=0.7129
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 23/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.61it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.74it/s]
📉 Train Loss=0.5829 | Val Loss=0.5724 | 🎯 Dice=0.2563 | 📈 IoU=0.4927 | 🔍 Prec=0.6418 | 🧠 Rec=0.6812
✅ Saved new best model.

🔁 Epoch 24/25
Training: 100%|██████████| 15/15 [00:09<00:00,  1.62it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.19it/s]
📉 Train Loss=0.5812 | Val Loss=0.5725 | 🎯 Dice=0.2634 | 📈 IoU=0.4795 | 🔍 Prec=0.5955 | 🧠 Rec=0.7128
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 25/25
Training: 100%|██████████| 15/15 [00:10<00:00,  1.49it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.93it/s]
[I 2025-07-17 16:22:28,226] Trial 0 finished with value: 0.4927300214767456 and parameters: {'batch_size': 2, 'lr': 0.00010256691315437255, 'weight_decay': 4.207053950287931e-07, 'accumulation_steps': 2}. Best is trial 0 with value: 0.4927300214767456.
📉 Train Loss=0.5780 | Val Loss=0.5712 | 🎯 Dice=0.2563 | 📈 IoU=0.4834 | 🔍 Prec=0.6336 | 🧠 Rec=0.6728
⚠️ No improvement. wait_counter=2/12

➤ Starting training: model=unet, backbone=None, bs=4, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.41it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.38it/s]
📉 Train Loss=0.7595 | Val Loss=0.7200 | 🎯 Dice=0.1498 | 📈 IoU=0.0621 | 🔍 Prec=0.2528 | 🧠 Rec=0.0763
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.83it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.22it/s]
📉 Train Loss=0.7207 | Val Loss=0.7019 | 🎯 Dice=0.1452 | 📈 IoU=0.0313 | 🔍 Prec=0.2241 | 🧠 Rec=0.0351
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 3/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.47it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
📉 Train Loss=0.7065 | Val Loss=0.6913 | 🎯 Dice=0.1430 | 📈 IoU=0.0288 | 🔍 Prec=0.2581 | 🧠 Rec=0.0312
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 8/8 [00:03<00:00,  2.51it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.28it/s]
📉 Train Loss=0.6945 | Val Loss=0.6841 | 🎯 Dice=0.1443 | 📈 IoU=0.0534 | 🔍 Prec=0.3901 | 🧠 Rec=0.0582
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 5/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.93it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.63it/s]
📉 Train Loss=0.6893 | Val Loss=0.6785 | 🎯 Dice=0.1477 | 📈 IoU=0.0753 | 🔍 Prec=0.4428 | 🧠 Rec=0.0835
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.41it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.13it/s]
📉 Train Loss=0.6816 | Val Loss=0.6738 | 🎯 Dice=0.1506 | 📈 IoU=0.1028 | 🔍 Prec=0.5385 | 🧠 Rec=0.1130
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.33it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.05it/s]
📉 Train Loss=0.6760 | Val Loss=0.6691 | 🎯 Dice=0.1548 | 📈 IoU=0.1517 | 🔍 Prec=0.4995 | 🧠 Rec=0.1792
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.90it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.42it/s]
📉 Train Loss=0.6735 | Val Loss=0.6649 | 🎯 Dice=0.1587 | 📈 IoU=0.1808 | 🔍 Prec=0.4792 | 🧠 Rec=0.2263
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.53it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.25it/s]
📉 Train Loss=0.6686 | Val Loss=0.6607 | 🎯 Dice=0.1618 | 📈 IoU=0.2006 | 🔍 Prec=0.4975 | 🧠 Rec=0.2538
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.39it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.12it/s]
📉 Train Loss=0.6656 | Val Loss=0.6560 | 🎯 Dice=0.1685 | 📈 IoU=0.2363 | 🔍 Prec=0.4984 | 🧠 Rec=0.3126
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.98it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.39it/s]
📉 Train Loss=0.6607 | Val Loss=0.6508 | 🎯 Dice=0.1756 | 📈 IoU=0.2657 | 🔍 Prec=0.5085 | 🧠 Rec=0.3596
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.40it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.13it/s]
📉 Train Loss=0.6576 | Val Loss=0.6465 | 🎯 Dice=0.1818 | 📈 IoU=0.2906 | 🔍 Prec=0.5131 | 🧠 Rec=0.4028
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.47it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.20it/s]
📉 Train Loss=0.6519 | Val Loss=0.6420 | 🎯 Dice=0.1874 | 📈 IoU=0.3136 | 🔍 Prec=0.5312 | 🧠 Rec=0.4351
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.99it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.41it/s]
📉 Train Loss=0.6483 | Val Loss=0.6378 | 🎯 Dice=0.1938 | 📈 IoU=0.3277 | 🔍 Prec=0.5305 | 🧠 Rec=0.4630
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.37it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.12it/s]
📉 Train Loss=0.6419 | Val Loss=0.6343 | 🎯 Dice=0.1997 | 📈 IoU=0.3411 | 🔍 Prec=0.5290 | 🧠 Rec=0.4916
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.42it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.04it/s]
📉 Train Loss=0.6394 | Val Loss=0.6303 | 🎯 Dice=0.2061 | 📈 IoU=0.3551 | 🔍 Prec=0.5263 | 🧠 Rec=0.5241
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.08it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.44it/s]
📉 Train Loss=0.6346 | Val Loss=0.6269 | 🎯 Dice=0.2096 | 📈 IoU=0.3681 | 🔍 Prec=0.5350 | 🧠 Rec=0.5435
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.28it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.21it/s]
📉 Train Loss=0.6331 | Val Loss=0.6242 | 🎯 Dice=0.2115 | 📈 IoU=0.3791 | 🔍 Prec=0.5474 | 🧠 Rec=0.5541
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.41it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.29it/s]
📉 Train Loss=0.6314 | Val Loss=0.6211 | 🎯 Dice=0.2154 | 📈 IoU=0.3898 | 🔍 Prec=0.5527 | 🧠 Rec=0.5716
✅ Saved new best model.

🔁 Epoch 20/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.23it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.53it/s]
📉 Train Loss=0.6298 | Val Loss=0.6179 | 🎯 Dice=0.2200 | 📈 IoU=0.4008 | 🔍 Prec=0.5547 | 🧠 Rec=0.5932
✅ Saved new best model.

🔁 Epoch 21/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.09it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.6249 | Val Loss=0.6158 | 🎯 Dice=0.2224 | 📈 IoU=0.4069 | 🔍 Prec=0.5594 | 🧠 Rec=0.6010
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.35it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.19it/s]
📉 Train Loss=0.6235 | Val Loss=0.6144 | 🎯 Dice=0.2220 | 📈 IoU=0.4108 | 🔍 Prec=0.5729 | 🧠 Rec=0.5938
✅ Saved new best model.

🔁 Epoch 23/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.48it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.37it/s]
📉 Train Loss=0.6219 | Val Loss=0.6127 | 🎯 Dice=0.2255 | 📈 IoU=0.4140 | 🔍 Prec=0.5658 | 🧠 Rec=0.6086
✅ Saved new best model.

🔁 Epoch 24/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.01it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.07it/s]
📉 Train Loss=0.6188 | Val Loss=0.6114 | 🎯 Dice=0.2292 | 📈 IoU=0.4158 | 🔍 Prec=0.5553 | 🧠 Rec=0.6260
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 8/8 [00:03<00:00,  2.43it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.17it/s]
[I 2025-07-17 16:24:17,158] Trial 1 finished with value: 0.42223307490348816 and parameters: {'batch_size': 4, 'lr': 0.0001329377199163636, 'weight_decay': 5.337032762603952e-07, 'accumulation_steps': 4}. Best is trial 0 with value: 0.4927300214767456.
📉 Train Loss=0.6150 | Val Loss=0.6087 | 🎯 Dice=0.2328 | 📈 IoU=0.4222 | 🔍 Prec=0.5571 | 🧠 Rec=0.6380
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=4, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.25it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.60it/s]
📉 Train Loss=0.7365 | Val Loss=0.6977 | 🎯 Dice=0.1428 | 📈 IoU=0.0126 | 🔍 Prec=0.1542 | 🧠 Rec=0.0134
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.22it/s]
📉 Train Loss=0.6946 | Val Loss=0.6790 | 🎯 Dice=0.1467 | 📈 IoU=0.0699 | 🔍 Prec=0.4606 | 🧠 Rec=0.0764
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.34it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.49it/s]
📉 Train Loss=0.6788 | Val Loss=0.6686 | 🎯 Dice=0.1524 | 📈 IoU=0.1415 | 🔍 Prec=0.5587 | 🧠 Rec=0.1598
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 8/8 [00:05<00:00,  1.56it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.09it/s]
📉 Train Loss=0.6690 | Val Loss=0.6587 | 🎯 Dice=0.1620 | 📈 IoU=0.2192 | 🔍 Prec=0.4901 | 🧠 Rec=0.2855
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.47it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.48it/s]
📉 Train Loss=0.6574 | Val Loss=0.6469 | 🎯 Dice=0.1781 | 📈 IoU=0.3014 | 🔍 Prec=0.5178 | 🧠 Rec=0.4207
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.40it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.6476 | Val Loss=0.6364 | 🎯 Dice=0.1942 | 📈 IoU=0.3450 | 🔍 Prec=0.5124 | 🧠 Rec=0.5154
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.57it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.70it/s]
📉 Train Loss=0.6388 | Val Loss=0.6275 | 🎯 Dice=0.2044 | 📈 IoU=0.3639 | 🔍 Prec=0.5405 | 🧠 Rec=0.5286
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
📉 Train Loss=0.6338 | Val Loss=0.6212 | 🎯 Dice=0.2139 | 📈 IoU=0.3868 | 🔍 Prec=0.5434 | 🧠 Rec=0.5757
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.52it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.24it/s]
📉 Train Loss=0.6245 | Val Loss=0.6169 | 🎯 Dice=0.2189 | 📈 IoU=0.3916 | 🔍 Prec=0.5443 | 🧠 Rec=0.5849
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.24it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.21it/s]
📉 Train Loss=0.6220 | Val Loss=0.6092 | 🎯 Dice=0.2271 | 📈 IoU=0.4126 | 🔍 Prec=0.5553 | 🧠 Rec=0.6180
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.60it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.6160 | Val Loss=0.6048 | 🎯 Dice=0.2307 | 📈 IoU=0.4271 | 🔍 Prec=0.5748 | 🧠 Rec=0.6264
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.23it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.13it/s]
📉 Train Loss=0.6124 | Val Loss=0.6008 | 🎯 Dice=0.2371 | 📈 IoU=0.4362 | 🔍 Prec=0.5710 | 🧠 Rec=0.6512
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.57it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.88it/s]
📉 Train Loss=0.6100 | Val Loss=0.6014 | 🎯 Dice=0.2364 | 📈 IoU=0.4260 | 🔍 Prec=0.5629 | 🧠 Rec=0.6390
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 14/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.27it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.19it/s]
📉 Train Loss=0.6068 | Val Loss=0.5975 | 🎯 Dice=0.2380 | 📈 IoU=0.4377 | 🔍 Prec=0.5848 | 🧠 Rec=0.6371
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.57it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.28it/s]
📉 Train Loss=0.6007 | Val Loss=0.5919 | 🎯 Dice=0.2442 | 📈 IoU=0.4553 | 🔍 Prec=0.5973 | 🧠 Rec=0.6593
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.25it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.10it/s]
📉 Train Loss=0.5982 | Val Loss=0.5904 | 🎯 Dice=0.2478 | 📈 IoU=0.4546 | 🔍 Prec=0.5832 | 🧠 Rec=0.6757
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 17/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.58it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.18it/s]
📉 Train Loss=0.5983 | Val Loss=0.5912 | 🎯 Dice=0.2428 | 📈 IoU=0.4453 | 🔍 Prec=0.5930 | 🧠 Rec=0.6434
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 18/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.30it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.36it/s]
📉 Train Loss=0.5947 | Val Loss=0.5858 | 🎯 Dice=0.2514 | 📈 IoU=0.4601 | 🔍 Prec=0.5857 | 🧠 Rec=0.6838
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.06it/s]
📉 Train Loss=0.5914 | Val Loss=0.5811 | 🎯 Dice=0.2578 | 📈 IoU=0.4759 | 🔍 Prec=0.5897 | 🧠 Rec=0.7129
✅ Saved new best model.

🔁 Epoch 20/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.35it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.46it/s]
📉 Train Loss=0.5937 | Val Loss=0.5784 | 🎯 Dice=0.2547 | 📈 IoU=0.4883 | 🔍 Prec=0.6295 | 🧠 Rec=0.6869
✅ Saved new best model.

🔁 Epoch 21/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.53it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.12it/s]
📉 Train Loss=0.5862 | Val Loss=0.5814 | 🎯 Dice=0.2524 | 📈 IoU=0.4655 | 🔍 Prec=0.6063 | 🧠 Rec=0.6688
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 22/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.39it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.59it/s]
📉 Train Loss=0.5820 | Val Loss=0.5803 | 🎯 Dice=0.2560 | 📈 IoU=0.4642 | 🔍 Prec=0.5928 | 🧠 Rec=0.6831
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 23/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.40it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.19it/s]
📉 Train Loss=0.5860 | Val Loss=0.5738 | 🎯 Dice=0.2613 | 📈 IoU=0.4879 | 🔍 Prec=0.6170 | 🧠 Rec=0.7015
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 24/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.57it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.42it/s]
📉 Train Loss=0.5829 | Val Loss=0.5710 | 🎯 Dice=0.2603 | 📈 IoU=0.4991 | 🔍 Prec=0.6391 | 🧠 Rec=0.6964
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.00it/s]
[I 2025-07-17 16:26:58,182] Trial 2 finished with value: 0.4990532398223877 and parameters: {'batch_size': 4, 'lr': 0.0001919814649902086, 'weight_decay': 2.9204338471814074e-06, 'accumulation_steps': 2}. Best is trial 2 with value: 0.4990532398223877.
📉 Train Loss=0.5792 | Val Loss=0.5710 | 🎯 Dice=0.2643 | 📈 IoU=0.4871 | 🔍 Prec=0.6082 | 🧠 Rec=0.7113
⚠️ No improvement. wait_counter=1/12

➤ Starting training: model=unet, backbone=None, bs=8, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.23it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.54it/s]
📉 Train Loss=0.7603 | Val Loss=0.7228 | 🎯 Dice=0.1459 | 📈 IoU=0.0732 | 🔍 Prec=0.2262 | 🧠 Rec=0.0980
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.37it/s]
📉 Train Loss=0.7248 | Val Loss=0.7062 | 🎯 Dice=0.1415 | 📈 IoU=0.0327 | 🔍 Prec=0.2380 | 🧠 Rec=0.0363
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 3/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.18it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.54it/s]
📉 Train Loss=0.7087 | Val Loss=0.6947 | 🎯 Dice=0.1388 | 📈 IoU=0.0189 | 🔍 Prec=0.2951 | 🧠 Rec=0.0197
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 4/4 [00:03<00:00,  1.26it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.57it/s]
📉 Train Loss=0.6983 | Val Loss=0.6877 | 🎯 Dice=0.1382 | 📈 IoU=0.0237 | 🔍 Prec=0.3534 | 🧠 Rec=0.0246
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 5/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.18it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.36it/s]
📉 Train Loss=0.6914 | Val Loss=0.6823 | 🎯 Dice=0.1395 | 📈 IoU=0.0441 | 🔍 Prec=0.4916 | 🧠 Rec=0.0464
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 6/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.06it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.61it/s]
📉 Train Loss=0.6879 | Val Loss=0.6772 | 🎯 Dice=0.1432 | 📈 IoU=0.0813 | 🔍 Prec=0.4952 | 🧠 Rec=0.0891
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.26it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.54it/s]
📉 Train Loss=0.6822 | Val Loss=0.6730 | 🎯 Dice=0.1482 | 📈 IoU=0.1247 | 🔍 Prec=0.4779 | 🧠 Rec=0.1450
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.22it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.47it/s]
📉 Train Loss=0.6771 | Val Loss=0.6696 | 🎯 Dice=0.1505 | 📈 IoU=0.1636 | 🔍 Prec=0.5053 | 🧠 Rec=0.1956
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.03s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.56it/s]
📉 Train Loss=0.6717 | Val Loss=0.6667 | 🎯 Dice=0.1532 | 📈 IoU=0.1892 | 🔍 Prec=0.4878 | 🧠 Rec=0.2382
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.22it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.67it/s]
📉 Train Loss=0.6688 | Val Loss=0.6635 | 🎯 Dice=0.1574 | 📈 IoU=0.2096 | 🔍 Prec=0.4617 | 🧠 Rec=0.2803
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.26it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.58it/s]
📉 Train Loss=0.6668 | Val Loss=0.6602 | 🎯 Dice=0.1597 | 📈 IoU=0.2244 | 🔍 Prec=0.4838 | 🧠 Rec=0.2984
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.00s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.62it/s]
📉 Train Loss=0.6646 | Val Loss=0.6557 | 🎯 Dice=0.1655 | 📈 IoU=0.2473 | 🔍 Prec=0.5011 | 🧠 Rec=0.3309
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.26it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.66it/s]
📉 Train Loss=0.6581 | Val Loss=0.6519 | 🎯 Dice=0.1732 | 📈 IoU=0.2673 | 🔍 Prec=0.4887 | 🧠 Rec=0.3732
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.25it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.65it/s]
📉 Train Loss=0.6544 | Val Loss=0.6479 | 🎯 Dice=0.1801 | 📈 IoU=0.2930 | 🔍 Prec=0.4862 | 🧠 Rec=0.4257
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.12s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.58it/s]
📉 Train Loss=0.6504 | Val Loss=0.6440 | 🎯 Dice=0.1850 | 📈 IoU=0.3101 | 🔍 Prec=0.4959 | 🧠 Rec=0.4541
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.30it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.70it/s]
📉 Train Loss=0.6470 | Val Loss=0.6403 | 🎯 Dice=0.1884 | 📈 IoU=0.3258 | 🔍 Prec=0.5152 | 🧠 Rec=0.4709
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.28it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.60it/s]
📉 Train Loss=0.6436 | Val Loss=0.6369 | 🎯 Dice=0.1927 | 📈 IoU=0.3428 | 🔍 Prec=0.5236 | 🧠 Rec=0.4993
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.01s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.28it/s]
📉 Train Loss=0.6425 | Val Loss=0.6334 | 🎯 Dice=0.1986 | 📈 IoU=0.3574 | 🔍 Prec=0.5204 | 🧠 Rec=0.5341
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.23it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.57it/s]
📉 Train Loss=0.6358 | Val Loss=0.6303 | 🎯 Dice=0.2040 | 📈 IoU=0.3671 | 🔍 Prec=0.5162 | 🧠 Rec=0.5610
✅ Saved new best model.

🔁 Epoch 20/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.25it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.67it/s]
📉 Train Loss=0.6362 | Val Loss=0.6278 | 🎯 Dice=0.2072 | 📈 IoU=0.3782 | 🔍 Prec=0.5214 | 🧠 Rec=0.5810
✅ Saved new best model.

🔁 Epoch 21/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.04s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.21it/s]
📉 Train Loss=0.6325 | Val Loss=0.6256 | 🎯 Dice=0.2091 | 📈 IoU=0.3858 | 🔍 Prec=0.5300 | 🧠 Rec=0.5880
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.24it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.63it/s]
📉 Train Loss=0.6286 | Val Loss=0.6237 | 🎯 Dice=0.2104 | 📈 IoU=0.3890 | 🔍 Prec=0.5374 | 🧠 Rec=0.5862
✅ Saved new best model.

🔁 Epoch 23/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.27it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.60it/s]
📉 Train Loss=0.6285 | Val Loss=0.6220 | 🎯 Dice=0.2131 | 📈 IoU=0.3948 | 🔍 Prec=0.5355 | 🧠 Rec=0.6020
✅ Saved new best model.

🔁 Epoch 24/25
Training: 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.30it/s]
📉 Train Loss=0.6235 | Val Loss=0.6201 | 🎯 Dice=0.2162 | 📈 IoU=0.4002 | 🔍 Prec=0.5337 | 🧠 Rec=0.6172
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 4/4 [00:03<00:00,  1.17it/s]
Validating: 100%|██████████| 1/1 [00:00<00:00,  1.60it/s]
[I 2025-07-17 16:28:46,481] Trial 3 finished with value: 0.40471982955932617 and parameters: {'batch_size': 8, 'lr': 0.0001096524277832185, 'weight_decay': 1.8205657658407247e-07, 'accumulation_steps': 2}. Best is trial 2 with value: 0.4990532398223877.
📉 Train Loss=0.6250 | Val Loss=0.6179 | 🎯 Dice=0.2190 | 📈 IoU=0.4047 | 🔍 Prec=0.5342 | 🧠 Rec=0.6273
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=4, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.53it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.04it/s]
📉 Train Loss=0.7473 | Val Loss=0.7122 | 🎯 Dice=0.1474 | 📈 IoU=0.0397 | 🔍 Prec=0.2148 | 🧠 Rec=0.0463
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.23it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.10it/s]
📉 Train Loss=0.7120 | Val Loss=0.6951 | 🎯 Dice=0.1430 | 📈 IoU=0.0229 | 🔍 Prec=0.2272 | 🧠 Rec=0.0246
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 3/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.56it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.6974 | Val Loss=0.6852 | 🎯 Dice=0.1423 | 📈 IoU=0.0341 | 🔍 Prec=0.3780 | 🧠 Rec=0.0362
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 8/8 [00:06<00:00,  1.27it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
📉 Train Loss=0.6872 | Val Loss=0.6779 | 🎯 Dice=0.1451 | 📈 IoU=0.0552 | 🔍 Prec=0.4709 | 🧠 Rec=0.0590
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.18it/s]
📉 Train Loss=0.6813 | Val Loss=0.6724 | 🎯 Dice=0.1502 | 📈 IoU=0.0885 | 🔍 Prec=0.4694 | 🧠 Rec=0.0986
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.22it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
📉 Train Loss=0.6771 | Val Loss=0.6672 | 🎯 Dice=0.1539 | 📈 IoU=0.1154 | 🔍 Prec=0.5008 | 🧠 Rec=0.1307
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.56it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.26it/s]
📉 Train Loss=0.6693 | Val Loss=0.6616 | 🎯 Dice=0.1595 | 📈 IoU=0.1664 | 🔍 Prec=0.4937 | 🧠 Rec=0.2015
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.24it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.20it/s]
📉 Train Loss=0.6652 | Val Loss=0.6561 | 🎯 Dice=0.1674 | 📈 IoU=0.2253 | 🔍 Prec=0.4788 | 🧠 Rec=0.3001
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.17it/s]
📉 Train Loss=0.6596 | Val Loss=0.6496 | 🎯 Dice=0.1763 | 📈 IoU=0.2791 | 🔍 Prec=0.4985 | 🧠 Rec=0.3893
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.26it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.67it/s]
📉 Train Loss=0.6542 | Val Loss=0.6429 | 🎯 Dice=0.1870 | 📈 IoU=0.3111 | 🔍 Prec=0.5007 | 🧠 Rec=0.4523
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.24it/s]
📉 Train Loss=0.6474 | Val Loss=0.6372 | 🎯 Dice=0.1932 | 📈 IoU=0.3334 | 🔍 Prec=0.5268 | 🧠 Rec=0.4772
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.31it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.44it/s]
📉 Train Loss=0.6411 | Val Loss=0.6332 | 🎯 Dice=0.1995 | 📈 IoU=0.3466 | 🔍 Prec=0.5200 | 🧠 Rec=0.5116
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.53it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.19it/s]
📉 Train Loss=0.6369 | Val Loss=0.6275 | 🎯 Dice=0.2050 | 📈 IoU=0.3745 | 🔍 Prec=0.5525 | 🧠 Rec=0.5395
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.38it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.39it/s]
📉 Train Loss=0.6370 | Val Loss=0.6226 | 🎯 Dice=0.2124 | 📈 IoU=0.3896 | 🔍 Prec=0.5458 | 🧠 Rec=0.5784
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.48it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.04it/s]
📉 Train Loss=0.6294 | Val Loss=0.6186 | 🎯 Dice=0.2161 | 📈 IoU=0.3994 | 🔍 Prec=0.5621 | 🧠 Rec=0.5815
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.58it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.52it/s]
📉 Train Loss=0.6265 | Val Loss=0.6149 | 🎯 Dice=0.2221 | 📈 IoU=0.4110 | 🔍 Prec=0.5603 | 🧠 Rec=0.6092
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.35it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.13it/s]
📉 Train Loss=0.6239 | Val Loss=0.6116 | 🎯 Dice=0.2270 | 📈 IoU=0.4192 | 🔍 Prec=0.5609 | 🧠 Rec=0.6265
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.53it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.12it/s]
📉 Train Loss=0.6206 | Val Loss=0.6089 | 🎯 Dice=0.2284 | 📈 IoU=0.4289 | 🔍 Prec=0.5776 | 🧠 Rec=0.6272
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.24it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.11it/s]
📉 Train Loss=0.6161 | Val Loss=0.6074 | 🎯 Dice=0.2320 | 📈 IoU=0.4254 | 🔍 Prec=0.5633 | 🧠 Rec=0.6373
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 20/25
Training: 100%|██████████| 8/8 [00:04<00:00,  1.61it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.94it/s]
📉 Train Loss=0.6156 | Val Loss=0.6061 | 🎯 Dice=0.2346 | 📈 IoU=0.4289 | 🔍 Prec=0.5633 | 🧠 Rec=0.6452
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 21/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.26it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.03it/s]
📉 Train Loss=0.6123 | Val Loss=0.6038 | 🎯 Dice=0.2339 | 📈 IoU=0.4364 | 🔍 Prec=0.5838 | 🧠 Rec=0.6362
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.57it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.14it/s]
📉 Train Loss=0.6100 | Val Loss=0.6019 | 🎯 Dice=0.2346 | 📈 IoU=0.4413 | 🔍 Prec=0.5929 | 🧠 Rec=0.6354
✅ Saved new best model.

🔁 Epoch 23/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.27it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
📉 Train Loss=0.6090 | Val Loss=0.6002 | 🎯 Dice=0.2414 | 📈 IoU=0.4422 | 🔍 Prec=0.5660 | 🧠 Rec=0.6716
✅ Saved new best model.

🔁 Epoch 24/25
Training: 100%|██████████| 8/8 [00:05<00:00,  1.55it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.6086 | Val Loss=0.6003 | 🎯 Dice=0.2385 | 📈 IoU=0.4404 | 🔍 Prec=0.5779 | 🧠 Rec=0.6514
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 25/25
Training: 100%|██████████| 8/8 [00:06<00:00,  1.27it/s]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.95it/s]
[I 2025-07-17 16:31:28,443] Trial 4 finished with value: 0.4429989457130432 and parameters: {'batch_size': 4, 'lr': 8.770946743725407e-05, 'weight_decay': 9.565499215943809e-06, 'accumulation_steps': 2}. Best is trial 2 with value: 0.4990532398223877.
📉 Train Loss=0.6055 | Val Loss=0.5986 | 🎯 Dice=0.2395 | 📈 IoU=0.4430 | 🔍 Prec=0.5821 | 🧠 Rec=0.6515
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.74it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.27it/s]
📉 Train Loss=0.7114 | Val Loss=0.6792 | 🎯 Dice=0.1377 | 📈 IoU=0.0206 | 🔍 Prec=0.3307 | 🧠 Rec=0.0214
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.77it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.01it/s]
📉 Train Loss=0.6745 | Val Loss=0.6606 | 🎯 Dice=0.1515 | 📈 IoU=0.1240 | 🔍 Prec=0.5444 | 🧠 Rec=0.1389
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.68it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.30it/s]
📉 Train Loss=0.6556 | Val Loss=0.6432 | 🎯 Dice=0.1755 | 📈 IoU=0.2921 | 🔍 Prec=0.4827 | 🧠 Rec=0.4264
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:17<00:00,  1.75it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.73it/s]
📉 Train Loss=0.6366 | Val Loss=0.6247 | 🎯 Dice=0.2039 | 📈 IoU=0.3737 | 🔍 Prec=0.5219 | 🧠 Rec=0.5695
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.76it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.20it/s]
📉 Train Loss=0.6222 | Val Loss=0.6155 | 🎯 Dice=0.2160 | 📈 IoU=0.3904 | 🔍 Prec=0.5365 | 🧠 Rec=0.5908
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.78it/s]
📉 Train Loss=0.6121 | Val Loss=0.6031 | 🎯 Dice=0.2305 | 📈 IoU=0.4270 | 🔍 Prec=0.5533 | 🧠 Rec=0.6534
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:18<00:00,  1.65it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.52it/s]
📉 Train Loss=0.6081 | Val Loss=0.5959 | 🎯 Dice=0.2359 | 📈 IoU=0.4416 | 🔍 Prec=0.5750 | 🧠 Rec=0.6572
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.80it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.98it/s]
📉 Train Loss=0.6011 | Val Loss=0.5926 | 🎯 Dice=0.2445 | 📈 IoU=0.4402 | 🔍 Prec=0.5519 | 🧠 Rec=0.6870
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.82it/s]
📉 Train Loss=0.5940 | Val Loss=0.5854 | 🎯 Dice=0.2494 | 📈 IoU=0.4577 | 🔍 Prec=0.5677 | 🧠 Rec=0.7041
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.76it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.02it/s]
📉 Train Loss=0.5877 | Val Loss=0.5835 | 🎯 Dice=0.2484 | 📈 IoU=0.4535 | 🔍 Prec=0.5773 | 🧠 Rec=0.6806
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.72it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.77it/s]
📉 Train Loss=0.5853 | Val Loss=0.5787 | 🎯 Dice=0.2536 | 📈 IoU=0.4631 | 🔍 Prec=0.5850 | 🧠 Rec=0.6913
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.63it/s]
📉 Train Loss=0.5802 | Val Loss=0.5721 | 🎯 Dice=0.2558 | 📈 IoU=0.4791 | 🔍 Prec=0.6134 | 🧠 Rec=0.6879
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.79it/s]
📉 Train Loss=0.5764 | Val Loss=0.5683 | 🎯 Dice=0.2618 | 📈 IoU=0.4819 | 🔍 Prec=0.6019 | 🧠 Rec=0.7088
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:18<00:00,  1.66it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.08it/s]
📉 Train Loss=0.5744 | Val Loss=0.5681 | 🎯 Dice=0.2597 | 📈 IoU=0.4744 | 🔍 Prec=0.6040 | 🧠 Rec=0.6902
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.79it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.56it/s]
📉 Train Loss=0.5717 | Val Loss=0.5651 | 🎯 Dice=0.2626 | 📈 IoU=0.4743 | 🔍 Prec=0.5980 | 🧠 Rec=0.6976
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.30it/s]
📉 Train Loss=0.5682 | Val Loss=0.5624 | 🎯 Dice=0.2617 | 📈 IoU=0.4751 | 🔍 Prec=0.6081 | 🧠 Rec=0.6863
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.74it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.45it/s]
📉 Train Loss=0.5693 | Val Loss=0.5569 | 🎯 Dice=0.2659 | 📈 IoU=0.4945 | 🔍 Prec=0.6347 | 🧠 Rec=0.6927
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.69it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.49it/s]
📉 Train Loss=0.5648 | Val Loss=0.5574 | 🎯 Dice=0.2702 | 📈 IoU=0.4766 | 🔍 Prec=0.5983 | 🧠 Rec=0.7021
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.41it/s]
📉 Train Loss=0.5618 | Val Loss=0.5531 | 🎯 Dice=0.2739 | 📈 IoU=0.4853 | 🔍 Prec=0.6018 | 🧠 Rec=0.7161
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.76it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.18it/s]
📉 Train Loss=0.5553 | Val Loss=0.5521 | 🎯 Dice=0.2740 | 📈 IoU=0.4838 | 🔍 Prec=0.6074 | 🧠 Rec=0.7055
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.73it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.53it/s]
📉 Train Loss=0.5529 | Val Loss=0.5452 | 🎯 Dice=0.2719 | 📈 IoU=0.5087 | 🔍 Prec=0.6624 | 🧠 Rec=0.6882
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.70it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.23it/s]
📉 Train Loss=0.5537 | Val Loss=0.5445 | 🎯 Dice=0.2793 | 📈 IoU=0.4966 | 🔍 Prec=0.6239 | 🧠 Rec=0.7102
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.89it/s]
📉 Train Loss=0.5523 | Val Loss=0.5444 | 🎯 Dice=0.2762 | 📈 IoU=0.4904 | 🔍 Prec=0.6258 | 🧠 Rec=0.6954
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.75it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.66it/s]
📉 Train Loss=0.5506 | Val Loss=0.5365 | 🎯 Dice=0.2857 | 📈 IoU=0.5182 | 🔍 Prec=0.6408 | 🧠 Rec=0.7317
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:18<00:00,  1.63it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.57it/s]
[I 2025-07-17 16:38:53,549] Trial 5 finished with value: 0.5182001829147339 and parameters: {'batch_size': 1, 'lr': 0.0001171329052910203, 'weight_decay': 0.0007556810141274422, 'accumulation_steps': 2}. Best is trial 5 with value: 0.5182001829147339.
📉 Train Loss=0.5477 | Val Loss=0.5400 | 🎯 Dice=0.2788 | 📈 IoU=0.4958 | 🔍 Prec=0.6397 | 🧠 Rec=0.6892
⚠️ No improvement. wait_counter=1/12

➤ Starting training: model=unet, backbone=None, bs=2, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.93it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.57it/s]
📉 Train Loss=0.7526 | Val Loss=0.7201 | 🎯 Dice=0.1476 | 📈 IoU=0.0678 | 🔍 Prec=0.2423 | 🧠 Rec=0.0862
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.40it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.01it/s]
📉 Train Loss=0.7211 | Val Loss=0.7036 | 🎯 Dice=0.1423 | 📈 IoU=0.0178 | 🔍 Prec=0.1802 | 🧠 Rec=0.0193
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 3/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.95it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.82it/s]
📉 Train Loss=0.7057 | Val Loss=0.6927 | 🎯 Dice=0.1397 | 📈 IoU=0.0064 | 🔍 Prec=0.1641 | 🧠 Rec=0.0066
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 15/15 [00:05<00:00,  2.61it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.37it/s]
📉 Train Loss=0.6970 | Val Loss=0.6853 | 🎯 Dice=0.1400 | 📈 IoU=0.0160 | 🔍 Prec=0.2908 | 🧠 Rec=0.0165
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 5/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.71it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.70it/s]
📉 Train Loss=0.6887 | Val Loss=0.6794 | 🎯 Dice=0.1426 | 📈 IoU=0.0441 | 🔍 Prec=0.4480 | 🧠 Rec=0.0466
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 6/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.95it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.78it/s]
📉 Train Loss=0.6846 | Val Loss=0.6745 | 🎯 Dice=0.1459 | 📈 IoU=0.0637 | 🔍 Prec=0.4778 | 🧠 Rec=0.0686
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 7/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.44it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.84it/s]
📉 Train Loss=0.6793 | Val Loss=0.6705 | 🎯 Dice=0.1494 | 📈 IoU=0.0885 | 🔍 Prec=0.5008 | 🧠 Rec=0.0971
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.99it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.96it/s]
📉 Train Loss=0.6716 | Val Loss=0.6660 | 🎯 Dice=0.1545 | 📈 IoU=0.1465 | 🔍 Prec=0.5148 | 🧠 Rec=0.1701
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.39it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.98it/s]
📉 Train Loss=0.6693 | Val Loss=0.6620 | 🎯 Dice=0.1590 | 📈 IoU=0.2027 | 🔍 Prec=0.5197 | 🧠 Rec=0.2510
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.93it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.05it/s]
📉 Train Loss=0.6670 | Val Loss=0.6578 | 🎯 Dice=0.1633 | 📈 IoU=0.2392 | 🔍 Prec=0.5062 | 🧠 Rec=0.3154
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.33it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.75it/s]
📉 Train Loss=0.6640 | Val Loss=0.6541 | 🎯 Dice=0.1681 | 📈 IoU=0.2579 | 🔍 Prec=0.5027 | 🧠 Rec=0.3486
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.93it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.85it/s]
📉 Train Loss=0.6555 | Val Loss=0.6490 | 🎯 Dice=0.1771 | 📈 IoU=0.2904 | 🔍 Prec=0.4956 | 🧠 Rec=0.4137
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.31it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.64it/s]
📉 Train Loss=0.6521 | Val Loss=0.6444 | 🎯 Dice=0.1855 | 📈 IoU=0.3162 | 🔍 Prec=0.4877 | 🧠 Rec=0.4750
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.87it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.06it/s]
📉 Train Loss=0.6497 | Val Loss=0.6410 | 🎯 Dice=0.1873 | 📈 IoU=0.3173 | 🔍 Prec=0.5131 | 🧠 Rec=0.4553
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.47it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.38it/s]
📉 Train Loss=0.6441 | Val Loss=0.6377 | 🎯 Dice=0.1925 | 📈 IoU=0.3382 | 🔍 Prec=0.5205 | 🧠 Rec=0.4932
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.77it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.64it/s]
📉 Train Loss=0.6416 | Val Loss=0.6337 | 🎯 Dice=0.2006 | 📈 IoU=0.3532 | 🔍 Prec=0.5076 | 🧠 Rec=0.5393
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.44it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.35it/s]
📉 Train Loss=0.6385 | Val Loss=0.6311 | 🎯 Dice=0.2038 | 📈 IoU=0.3596 | 🔍 Prec=0.5064 | 🧠 Rec=0.5558
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.83it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.84it/s]
📉 Train Loss=0.6333 | Val Loss=0.6274 | 🎯 Dice=0.2067 | 📈 IoU=0.3751 | 🔍 Prec=0.5310 | 🧠 Rec=0.5629
✅ Saved new best model.

🔁 Epoch 19/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.72it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.49it/s]
📉 Train Loss=0.6306 | Val Loss=0.6235 | 🎯 Dice=0.2108 | 📈 IoU=0.3882 | 🔍 Prec=0.5442 | 🧠 Rec=0.5771
✅ Saved new best model.

🔁 Epoch 20/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.55it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.88it/s]
📉 Train Loss=0.6284 | Val Loss=0.6212 | 🎯 Dice=0.2158 | 📈 IoU=0.3952 | 🔍 Prec=0.5338 | 🧠 Rec=0.6057
✅ Saved new best model.

🔁 Epoch 21/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.89it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.70it/s]
📉 Train Loss=0.6300 | Val Loss=0.6184 | 🎯 Dice=0.2190 | 📈 IoU=0.4068 | 🔍 Prec=0.5454 | 🧠 Rec=0.6177
✅ Saved new best model.

🔁 Epoch 22/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.39it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.86it/s]
📉 Train Loss=0.6258 | Val Loss=0.6168 | 🎯 Dice=0.2194 | 📈 IoU=0.4087 | 🔍 Prec=0.5528 | 🧠 Rec=0.6130
✅ Saved new best model.

🔁 Epoch 23/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.91it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.65it/s]
📉 Train Loss=0.6212 | Val Loss=0.6146 | 🎯 Dice=0.2226 | 📈 IoU=0.4151 | 🔍 Prec=0.5533 | 🧠 Rec=0.6268
✅ Saved new best model.

🔁 Epoch 24/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.32it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.72it/s]
📉 Train Loss=0.6200 | Val Loss=0.6129 | 🎯 Dice=0.2259 | 📈 IoU=0.4180 | 🔍 Prec=0.5450 | 🧠 Rec=0.6445
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.93it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.80it/s]
[I 2025-07-17 16:41:30,840] Trial 6 finished with value: 0.424712210893631 and parameters: {'batch_size': 2, 'lr': 6.157785861833019e-05, 'weight_decay': 2.0013420622879973e-06, 'accumulation_steps': 4}. Best is trial 5 with value: 0.5182001829147339.
📉 Train Loss=0.6194 | Val Loss=0.6106 | 🎯 Dice=0.2273 | 📈 IoU=0.4247 | 🔍 Prec=0.5538 | 🧠 Rec=0.6482
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=4, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.28s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.22it/s]
📉 Train Loss=0.6895 | Val Loss=0.6550 | 🎯 Dice=0.1450 | 📈 IoU=0.0665 | 🔍 Prec=0.4353 | 🧠 Rec=0.0732
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.29s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.11it/s]
📉 Train Loss=0.6385 | Val Loss=0.6102 | 🎯 Dice=0.1929 | 📈 IoU=0.3143 | 🔍 Prec=0.5204 | 🧠 Rec=0.4439
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.22s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.38it/s]
📉 Train Loss=0.6015 | Val Loss=0.5743 | 🎯 Dice=0.2289 | 📈 IoU=0.3782 | 🔍 Prec=0.5579 | 🧠 Rec=0.5405
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 8/8 [00:09<00:00,  1.13s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.08it/s]
📉 Train Loss=0.5673 | Val Loss=0.5478 | 🎯 Dice=0.2652 | 📈 IoU=0.4422 | 🔍 Prec=0.5949 | 🧠 Rec=0.6342
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.28s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.15it/s]
📉 Train Loss=0.5509 | Val Loss=0.5243 | 🎯 Dice=0.2860 | 📈 IoU=0.4779 | 🔍 Prec=0.6461 | 🧠 Rec=0.6485
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.27s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.17it/s]
📉 Train Loss=0.5381 | Val Loss=0.5192 | 🎯 Dice=0.2971 | 📈 IoU=0.4679 | 🔍 Prec=0.6111 | 🧠 Rec=0.6671
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 7/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.29s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.09it/s]
📉 Train Loss=0.5204 | Val Loss=0.5130 | 🎯 Dice=0.3061 | 📈 IoU=0.4688 | 🔍 Prec=0.6145 | 🧠 Rec=0.6652
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 8/25
Training: 100%|██████████| 8/8 [00:08<00:00,  1.11s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.65it/s]
📉 Train Loss=0.5227 | Val Loss=0.5004 | 🎯 Dice=0.3122 | 📈 IoU=0.5000 | 🔍 Prec=0.6714 | 🧠 Rec=0.6627
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.23s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.05it/s]
📉 Train Loss=0.5120 | Val Loss=0.4985 | 🎯 Dice=0.3131 | 📈 IoU=0.5011 | 🔍 Prec=0.6856 | 🧠 Rec=0.6518
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.28s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.14it/s]
📉 Train Loss=0.5096 | Val Loss=0.5113 | 🎯 Dice=0.2964 | 📈 IoU=0.4592 | 🔍 Prec=0.6608 | 🧠 Rec=0.6019
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 11/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.24s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.07it/s]
📉 Train Loss=0.5104 | Val Loss=0.4966 | 🎯 Dice=0.3206 | 📈 IoU=0.4979 | 🔍 Prec=0.6630 | 🧠 Rec=0.6672
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 12/25
Training: 100%|██████████| 8/8 [00:08<00:00,  1.09s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.25it/s]
📉 Train Loss=0.5068 | Val Loss=0.4952 | 🎯 Dice=0.3333 | 📈 IoU=0.4981 | 🔍 Prec=0.6270 | 🧠 Rec=0.7091
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 13/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.20s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.12it/s]
📉 Train Loss=0.5040 | Val Loss=0.4932 | 🎯 Dice=0.3293 | 📈 IoU=0.5001 | 🔍 Prec=0.6512 | 🧠 Rec=0.6843
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 14/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.27s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.00it/s]
📉 Train Loss=0.5031 | Val Loss=0.4928 | 🎯 Dice=0.3192 | 📈 IoU=0.4993 | 🔍 Prec=0.7019 | 🧠 Rec=0.6348
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 15/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.25s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.10it/s]
📉 Train Loss=0.5082 | Val Loss=0.4930 | 🎯 Dice=0.3253 | 📈 IoU=0.4968 | 🔍 Prec=0.6680 | 🧠 Rec=0.6608
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 16/25
Training: 100%|██████████| 8/8 [00:08<00:00,  1.12s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.32it/s]
📉 Train Loss=0.5023 | Val Loss=0.4928 | 🎯 Dice=0.3243 | 📈 IoU=0.4961 | 🔍 Prec=0.6703 | 🧠 Rec=0.6573
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 17/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.20s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.10it/s]
📉 Train Loss=0.5036 | Val Loss=0.4993 | 🎯 Dice=0.3208 | 📈 IoU=0.4805 | 🔍 Prec=0.6544 | 🧠 Rec=0.6450
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 18/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.24s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.09it/s]
📉 Train Loss=0.5057 | Val Loss=0.4949 | 🎯 Dice=0.3238 | 📈 IoU=0.4893 | 🔍 Prec=0.6413 | 🧠 Rec=0.6748
⚠️ No improvement. wait_counter=9/12

🔁 Epoch 19/25
Training: 100%|██████████| 8/8 [00:10<00:00,  1.25s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.02it/s]
📉 Train Loss=0.5002 | Val Loss=0.4928 | 🎯 Dice=0.3271 | 📈 IoU=0.4979 | 🔍 Prec=0.6697 | 🧠 Rec=0.6612
⚠️ No improvement. wait_counter=10/12

🔁 Epoch 20/25
Training: 100%|██████████| 8/8 [00:08<00:00,  1.12s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  2.41it/s]
📉 Train Loss=0.4981 | Val Loss=0.5020 | 🎯 Dice=0.3206 | 📈 IoU=0.4700 | 🔍 Prec=0.6408 | 🧠 Rec=0.6394
⚠️ No improvement. wait_counter=11/12

🔁 Epoch 21/25
Training: 100%|██████████| 8/8 [00:09<00:00,  1.17s/it]
Validating: 100%|██████████| 2/2 [00:00<00:00,  3.16it/s]
[I 2025-07-17 16:45:11,142] Trial 7 finished with value: 0.5011325627565384 and parameters: {'batch_size': 4, 'lr': 0.002010777263345145, 'weight_decay': 1.9870215385428617e-07, 'accumulation_steps': 1}. Best is trial 5 with value: 0.5182001829147339.
📉 Train Loss=0.4931 | Val Loss=0.4921 | 🎯 Dice=0.3316 | 📈 IoU=0.4964 | 🔍 Prec=0.6513 | 🧠 Rec=0.6773
⚠️ No improvement. wait_counter=12/12
⏹ Early stopping triggered!

➤ Starting training: model=unet, backbone=None, bs=2, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.37it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.75it/s]
📉 Train Loss=0.7207 | Val Loss=0.6707 | 🎯 Dice=0.1375 | 📈 IoU=0.0694 | 🔍 Prec=0.3563 | 🧠 Rec=0.0795
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.93it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.02it/s]
📉 Train Loss=0.6664 | Val Loss=0.6474 | 🎯 Dice=0.1523 | 📈 IoU=0.1031 | 🔍 Prec=0.4456 | 🧠 Rec=0.1183
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.38it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.62it/s]
📉 Train Loss=0.6431 | Val Loss=0.6245 | 🎯 Dice=0.1785 | 📈 IoU=0.2755 | 🔍 Prec=0.4840 | 🧠 Rec=0.3942
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 15/15 [00:05<00:00,  2.88it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.58it/s]
📉 Train Loss=0.6205 | Val Loss=0.6076 | 🎯 Dice=0.1893 | 📈 IoU=0.3189 | 🔍 Prec=0.5948 | 🧠 Rec=0.4082
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.55it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.29it/s]
📉 Train Loss=0.6011 | Val Loss=0.5810 | 🎯 Dice=0.2424 | 📈 IoU=0.4001 | 🔍 Prec=0.5077 | 🧠 Rec=0.6548
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.77it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.70it/s]
📉 Train Loss=0.5811 | Val Loss=0.5684 | 🎯 Dice=0.2410 | 📈 IoU=0.4337 | 🔍 Prec=0.5908 | 🧠 Rec=0.6213
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.67it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.37it/s]
📉 Train Loss=0.5710 | Val Loss=0.5557 | 🎯 Dice=0.2560 | 📈 IoU=0.4427 | 🔍 Prec=0.5868 | 🧠 Rec=0.6441
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.57it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.83it/s]
📉 Train Loss=0.5555 | Val Loss=0.5504 | 🎯 Dice=0.2601 | 📈 IoU=0.4280 | 🔍 Prec=0.5974 | 🧠 Rec=0.6026
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 9/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.91it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.81it/s]
📉 Train Loss=0.5445 | Val Loss=0.5318 | 🎯 Dice=0.2886 | 📈 IoU=0.4785 | 🔍 Prec=0.5980 | 🧠 Rec=0.7064
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.36it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.83it/s]
📉 Train Loss=0.5400 | Val Loss=0.5268 | 🎯 Dice=0.2743 | 📈 IoU=0.4863 | 🔍 Prec=0.6761 | 🧠 Rec=0.6351
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.96it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.70it/s]
📉 Train Loss=0.5300 | Val Loss=0.5322 | 🎯 Dice=0.2818 | 📈 IoU=0.4430 | 🔍 Prec=0.5975 | 🧠 Rec=0.6326
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 12/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.34it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.11it/s]
📉 Train Loss=0.5290 | Val Loss=0.5272 | 🎯 Dice=0.2900 | 📈 IoU=0.4499 | 🔍 Prec=0.5975 | 🧠 Rec=0.6468
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 13/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.92it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.97it/s]
📉 Train Loss=0.5235 | Val Loss=0.5163 | 🎯 Dice=0.2934 | 📈 IoU=0.4759 | 🔍 Prec=0.6469 | 🧠 Rec=0.6440
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 14/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.37it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.91it/s]
📉 Train Loss=0.5274 | Val Loss=0.5027 | 🎯 Dice=0.3033 | 📈 IoU=0.5149 | 🔍 Prec=0.6911 | 🧠 Rec=0.6699
✅ Saved new best model.

🔁 Epoch 15/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.97it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.85it/s]
📉 Train Loss=0.5163 | Val Loss=0.5079 | 🎯 Dice=0.3072 | 📈 IoU=0.4851 | 🔍 Prec=0.6277 | 🧠 Rec=0.6823
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 16/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.45it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.40it/s]
📉 Train Loss=0.5134 | Val Loss=0.5121 | 🎯 Dice=0.2940 | 📈 IoU=0.4682 | 🔍 Prec=0.6545 | 🧠 Rec=0.6236
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 17/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.95it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.92it/s]
📉 Train Loss=0.5094 | Val Loss=0.5018 | 🎯 Dice=0.3136 | 📈 IoU=0.4936 | 🔍 Prec=0.6439 | 🧠 Rec=0.6804
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 18/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.80it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  4.42it/s]
📉 Train Loss=0.5069 | Val Loss=0.4991 | 🎯 Dice=0.3204 | 📈 IoU=0.4971 | 🔍 Prec=0.6246 | 🧠 Rec=0.7101
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 19/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.63it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.92it/s]
📉 Train Loss=0.5069 | Val Loss=0.4980 | 🎯 Dice=0.3110 | 📈 IoU=0.4965 | 🔍 Prec=0.6738 | 🧠 Rec=0.6550
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 20/25
Training: 100%|██████████| 15/15 [00:04<00:00,  3.01it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  6.16it/s]
📉 Train Loss=0.5122 | Val Loss=0.5008 | 🎯 Dice=0.3144 | 📈 IoU=0.4835 | 🔍 Prec=0.6346 | 🧠 Rec=0.6715
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 21/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.38it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.76it/s]
📉 Train Loss=0.5011 | Val Loss=0.4973 | 🎯 Dice=0.3151 | 📈 IoU=0.4940 | 🔍 Prec=0.6615 | 🧠 Rec=0.6627
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 22/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.90it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.74it/s]
📉 Train Loss=0.5054 | Val Loss=0.4991 | 🎯 Dice=0.3185 | 📈 IoU=0.4870 | 🔍 Prec=0.6325 | 🧠 Rec=0.6808
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 23/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.35it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.82it/s]
📉 Train Loss=0.5094 | Val Loss=0.5073 | 🎯 Dice=0.2973 | 📈 IoU=0.4593 | 🔍 Prec=0.6703 | 🧠 Rec=0.5948
⚠️ No improvement. wait_counter=9/12

🔁 Epoch 24/25
Training: 100%|██████████| 15/15 [00:05<00:00,  2.91it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.91it/s]
📉 Train Loss=0.5073 | Val Loss=0.4945 | 🎯 Dice=0.3246 | 📈 IoU=0.4976 | 🔍 Prec=0.6385 | 🧠 Rec=0.6943
⚠️ No improvement. wait_counter=10/12

🔁 Epoch 25/25
Training: 100%|██████████| 15/15 [00:06<00:00,  2.38it/s]
Validating: 100%|██████████| 3/3 [00:00<00:00,  5.90it/s]
[I 2025-07-17 16:47:48,298] Trial 8 finished with value: 0.5149485866228739 and parameters: {'batch_size': 2, 'lr': 0.001743856312272965, 'weight_decay': 1.9777828512462694e-07, 'accumulation_steps': 4}. Best is trial 5 with value: 0.5182001829147339.
📉 Train Loss=0.4942 | Val Loss=0.4965 | 🎯 Dice=0.3159 | 📈 IoU=0.4905 | 🔍 Prec=0.6645 | 🧠 Rec=0.6535
⚠️ No improvement. wait_counter=11/12

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.26it/s]
📉 Train Loss=0.7012 | Val Loss=0.6708 | 🎯 Dice=0.1403 | 📈 IoU=0.0301 | 🔍 Prec=0.5756 | 🧠 Rec=0.0308
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.79it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.17it/s]
📉 Train Loss=0.6635 | Val Loss=0.6501 | 🎯 Dice=0.1607 | 📈 IoU=0.2398 | 🔍 Prec=0.4801 | 🧠 Rec=0.3248
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.45it/s]
📉 Train Loss=0.6386 | Val Loss=0.6210 | 🎯 Dice=0.2006 | 📈 IoU=0.3614 | 🔍 Prec=0.5180 | 🧠 Rec=0.5465
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:17<00:00,  1.69it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.81it/s]
📉 Train Loss=0.6158 | Val Loss=0.6035 | 🎯 Dice=0.2272 | 📈 IoU=0.4129 | 🔍 Prec=0.5338 | 🧠 Rec=0.6477
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.79it/s]
📉 Train Loss=0.6040 | Val Loss=0.5904 | 🎯 Dice=0.2432 | 📈 IoU=0.4384 | 🔍 Prec=0.5511 | 🧠 Rec=0.6834
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.08it/s]
📉 Train Loss=0.5931 | Val Loss=0.5826 | 🎯 Dice=0.2503 | 📈 IoU=0.4474 | 🔍 Prec=0.5564 | 🧠 Rec=0.6967
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.71it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.72it/s]
📉 Train Loss=0.5804 | Val Loss=0.5734 | 🎯 Dice=0.2516 | 📈 IoU=0.4643 | 🔍 Prec=0.5950 | 🧠 Rec=0.6800
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.76it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.53it/s]
📉 Train Loss=0.5771 | Val Loss=0.5672 | 🎯 Dice=0.2564 | 📈 IoU=0.4685 | 🔍 Prec=0.6050 | 🧠 Rec=0.6765
✅ Saved new best model.

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.19it/s]
📉 Train Loss=0.5700 | Val Loss=0.5576 | 🎯 Dice=0.2639 | 📈 IoU=0.4922 | 🔍 Prec=0.6270 | 🧠 Rec=0.6971
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.80it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.05it/s]
📉 Train Loss=0.5660 | Val Loss=0.5609 | 🎯 Dice=0.2569 | 📈 IoU=0.4598 | 🔍 Prec=0.6065 | 🧠 Rec=0.6565
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.70it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.01it/s]
📉 Train Loss=0.5583 | Val Loss=0.5470 | 🎯 Dice=0.2710 | 📈 IoU=0.5055 | 🔍 Prec=0.6441 | 🧠 Rec=0.7026
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.80it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.72it/s]
📉 Train Loss=0.5554 | Val Loss=0.5529 | 🎯 Dice=0.2654 | 📈 IoU=0.4561 | 🔍 Prec=0.5990 | 🧠 Rec=0.6580
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.28it/s]
📉 Train Loss=0.5559 | Val Loss=0.5379 | 🎯 Dice=0.2849 | 📈 IoU=0.5095 | 🔍 Prec=0.6207 | 🧠 Rec=0.7409
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.13it/s]
📉 Train Loss=0.5459 | Val Loss=0.5417 | 🎯 Dice=0.2749 | 📈 IoU=0.4754 | 🔍 Prec=0.6214 | 🧠 Rec=0.6708
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.67it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.17it/s]
📉 Train Loss=0.5390 | Val Loss=0.5286 | 🎯 Dice=0.2893 | 📈 IoU=0.5197 | 🔍 Prec=0.6531 | 🧠 Rec=0.7191
✅ Saved new best model.

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.88it/s]
📉 Train Loss=0.5430 | Val Loss=0.5295 | 🎯 Dice=0.2821 | 📈 IoU=0.5015 | 🔍 Prec=0.6528 | 🧠 Rec=0.6858
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.39it/s]
📉 Train Loss=0.5372 | Val Loss=0.5233 | 🎯 Dice=0.2992 | 📈 IoU=0.5127 | 🔍 Prec=0.6243 | 🧠 Rec=0.7429
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.82it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.39it/s]
📉 Train Loss=0.5342 | Val Loss=0.5211 | 🎯 Dice=0.2981 | 📈 IoU=0.5115 | 🔍 Prec=0.6324 | 🧠 Rec=0.7293
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.68it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.88it/s]
📉 Train Loss=0.5278 | Val Loss=0.5241 | 🎯 Dice=0.2972 | 📈 IoU=0.4920 | 🔍 Prec=0.6087 | 🧠 Rec=0.7210
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.84it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.70it/s]
📉 Train Loss=0.5315 | Val Loss=0.5197 | 🎯 Dice=0.2976 | 📈 IoU=0.5010 | 🔍 Prec=0.6363 | 🧠 Rec=0.7035
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.81it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.23it/s]
📉 Train Loss=0.5292 | Val Loss=0.5204 | 🎯 Dice=0.2882 | 📈 IoU=0.4918 | 🔍 Prec=0.6564 | 🧠 Rec=0.6639
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.79it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.17it/s]
📉 Train Loss=0.5219 | Val Loss=0.5162 | 🎯 Dice=0.3047 | 📈 IoU=0.5000 | 🔍 Prec=0.6238 | 🧠 Rec=0.7171
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:17<00:00,  1.69it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.29it/s]
📉 Train Loss=0.5202 | Val Loss=0.5114 | 🎯 Dice=0.2997 | 📈 IoU=0.5108 | 🔍 Prec=0.6619 | 🧠 Rec=0.6926
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.83it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.89it/s]
📉 Train Loss=0.5173 | Val Loss=0.5097 | 🎯 Dice=0.3025 | 📈 IoU=0.5092 | 🔍 Prec=0.6550 | 🧠 Rec=0.6975
⚠️ No improvement. wait_counter=9/12

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:16<00:00,  1.78it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.06it/s]
[I 2025-07-17 16:55:04,318] Trial 9 finished with value: 0.5381693482398987 and parameters: {'batch_size': 1, 'lr': 0.00022353042733892474, 'weight_decay': 8.287522363768158e-05, 'accumulation_steps': 2}. Best is trial 9 with value: 0.5381693482398987.
📉 Train Loss=0.5113 | Val Loss=0.5005 | 🎯 Dice=0.3138 | 📈 IoU=0.5382 | 🔍 Prec=0.6727 | 🧠 Rec=0.7306
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:33<00:00,  1.11s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.12it/s]
📉 Train Loss=0.6671 | Val Loss=0.6161 | 🎯 Dice=0.2081 | 📈 IoU=0.3647 | 🔍 Prec=0.4969 | 🧠 Rec=0.5802
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.53it/s]
📉 Train Loss=0.6042 | Val Loss=0.5770 | 🎯 Dice=0.2441 | 📈 IoU=0.4385 | 🔍 Prec=0.5672 | 🧠 Rec=0.6602
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.58it/s]
📉 Train Loss=0.5743 | Val Loss=0.5574 | 🎯 Dice=0.2483 | 📈 IoU=0.4506 | 🔍 Prec=0.6484 | 🧠 Rec=0.5973
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:32<00:00,  1.09s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.97it/s]
📉 Train Loss=0.5549 | Val Loss=0.5365 | 🎯 Dice=0.2768 | 📈 IoU=0.4817 | 🔍 Prec=0.6270 | 🧠 Rec=0.6759
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.00it/s]
📉 Train Loss=0.5408 | Val Loss=0.5204 | 🎯 Dice=0.2868 | 📈 IoU=0.5038 | 🔍 Prec=0.6625 | 🧠 Rec=0.6790
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.48it/s]
📉 Train Loss=0.5229 | Val Loss=0.5166 | 🎯 Dice=0.2933 | 📈 IoU=0.4848 | 🔍 Prec=0.6311 | 🧠 Rec=0.6776
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.78it/s]
📉 Train Loss=0.5239 | Val Loss=0.5077 | 🎯 Dice=0.3021 | 📈 IoU=0.4997 | 🔍 Prec=0.6457 | 🧠 Rec=0.6898
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.33it/s]
📉 Train Loss=0.5157 | Val Loss=0.5085 | 🎯 Dice=0.2998 | 📈 IoU=0.4833 | 🔍 Prec=0.6381 | 🧠 Rec=0.6672
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.75it/s]
📉 Train Loss=0.5115 | Val Loss=0.5211 | 🎯 Dice=0.2755 | 📈 IoU=0.4395 | 🔍 Prec=0.6658 | 🧠 Rec=0.5654
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.75it/s]
📉 Train Loss=0.5101 | Val Loss=0.5089 | 🎯 Dice=0.2957 | 📈 IoU=0.4691 | 🔍 Prec=0.6586 | 🧠 Rec=0.6211
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.10s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.65it/s]
📉 Train Loss=0.5069 | Val Loss=0.5060 | 🎯 Dice=0.3128 | 📈 IoU=0.4714 | 🔍 Prec=0.6018 | 🧠 Rec=0.6867
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.93it/s]
📉 Train Loss=0.5070 | Val Loss=0.4858 | 🎯 Dice=0.3199 | 📈 IoU=0.5319 | 🔍 Prec=0.7086 | 🧠 Rec=0.6825
✅ Saved new best model.

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.67it/s]
📉 Train Loss=0.4934 | Val Loss=0.5085 | 🎯 Dice=0.2998 | 📈 IoU=0.4575 | 🔍 Prec=0.6278 | 🧠 Rec=0.6295
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.19it/s]
📉 Train Loss=0.4958 | Val Loss=0.4844 | 🎯 Dice=0.3224 | 📈 IoU=0.5263 | 🔍 Prec=0.7058 | 🧠 Rec=0.6752
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.98it/s]
📉 Train Loss=0.4912 | Val Loss=0.4815 | 🎯 Dice=0.3297 | 📈 IoU=0.5292 | 🔍 Prec=0.6852 | 🧠 Rec=0.7005
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.00s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.89it/s]
📉 Train Loss=0.5069 | Val Loss=0.4726 | 🎯 Dice=0.3394 | 📈 IoU=0.5625 | 🔍 Prec=0.7082 | 🧠 Rec=0.7339
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.00it/s]
📉 Train Loss=0.4983 | Val Loss=0.4750 | 🎯 Dice=0.3364 | 📈 IoU=0.5515 | 🔍 Prec=0.7026 | 🧠 Rec=0.7209
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.06it/s]
📉 Train Loss=0.4895 | Val Loss=0.4875 | 🎯 Dice=0.3283 | 📈 IoU=0.5092 | 🔍 Prec=0.6454 | 🧠 Rec=0.7084
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.71it/s]
📉 Train Loss=0.4931 | Val Loss=0.4948 | 🎯 Dice=0.3171 | 📈 IoU=0.4857 | 🔍 Prec=0.6570 | 🧠 Rec=0.6523
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.13it/s]
📉 Train Loss=0.4936 | Val Loss=0.4775 | 🎯 Dice=0.3347 | 📈 IoU=0.5361 | 🔍 Prec=0.7036 | 🧠 Rec=0.6940
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.49it/s]
📉 Train Loss=0.4810 | Val Loss=0.4911 | 🎯 Dice=0.3257 | 📈 IoU=0.4944 | 🔍 Prec=0.6527 | 🧠 Rec=0.6729
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.05it/s]
📉 Train Loss=0.4844 | Val Loss=0.4822 | 🎯 Dice=0.3404 | 📈 IoU=0.5164 | 🔍 Prec=0.6450 | 🧠 Rec=0.7228
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.66it/s]
📉 Train Loss=0.4846 | Val Loss=0.4920 | 🎯 Dice=0.3206 | 📈 IoU=0.4886 | 🔍 Prec=0.6488 | 🧠 Rec=0.6661
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.22it/s]
📉 Train Loss=0.4880 | Val Loss=0.4761 | 🎯 Dice=0.3412 | 📈 IoU=0.5359 | 🔍 Prec=0.6746 | 🧠 Rec=0.7241
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.56it/s]
[I 2025-07-17 17:08:19,540] Trial 10 finished with value: 0.5786715745925903 and parameters: {'batch_size': 1, 'lr': 0.00046533520421806487, 'weight_decay': 0.00025935129154281333, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.4795 | Val Loss=0.4635 | 🎯 Dice=0.3538 | 📈 IoU=0.5787 | 🔍 Prec=0.7110 | 🧠 Rec=0.7580
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.07it/s]
📉 Train Loss=0.6675 | Val Loss=0.6288 | 🎯 Dice=0.1881 | 📈 IoU=0.3146 | 🔍 Prec=0.4812 | 🧠 Rec=0.4764
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.08s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.11it/s]
📉 Train Loss=0.6029 | Val Loss=0.5745 | 🎯 Dice=0.2437 | 📈 IoU=0.4478 | 🔍 Prec=0.5872 | 🧠 Rec=0.6549
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.01it/s]
📉 Train Loss=0.5720 | Val Loss=0.5536 | 🎯 Dice=0.2527 | 📈 IoU=0.4723 | 🔍 Prec=0.6546 | 🧠 Rec=0.6304
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.19it/s]
📉 Train Loss=0.5548 | Val Loss=0.5372 | 🎯 Dice=0.2784 | 📈 IoU=0.4776 | 🔍 Prec=0.6089 | 🧠 Rec=0.6903
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.09s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.78it/s]
📉 Train Loss=0.5332 | Val Loss=0.5370 | 🎯 Dice=0.2760 | 📈 IoU=0.4412 | 🔍 Prec=0.5970 | 🧠 Rec=0.6296
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.10it/s]
📉 Train Loss=0.5272 | Val Loss=0.5087 | 🎯 Dice=0.2965 | 📈 IoU=0.5153 | 🔍 Prec=0.6910 | 🧠 Rec=0.6709
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.56it/s]
📉 Train Loss=0.5150 | Val Loss=0.5079 | 🎯 Dice=0.2989 | 📈 IoU=0.5016 | 🔍 Prec=0.6593 | 🧠 Rec=0.6785
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.90it/s]
📉 Train Loss=0.5113 | Val Loss=0.5030 | 🎯 Dice=0.2928 | 📈 IoU=0.5101 | 🔍 Prec=0.7304 | 🧠 Rec=0.6297
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.15it/s]
📉 Train Loss=0.5049 | Val Loss=0.5127 | 🎯 Dice=0.2842 | 📈 IoU=0.4650 | 🔍 Prec=0.6981 | 🧠 Rec=0.5835
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.18it/s]
📉 Train Loss=0.5025 | Val Loss=0.5016 | 🎯 Dice=0.3054 | 📈 IoU=0.4870 | 🔍 Prec=0.6738 | 🧠 Rec=0.6386
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.22it/s]
📉 Train Loss=0.4985 | Val Loss=0.5035 | 🎯 Dice=0.3096 | 📈 IoU=0.4750 | 🔍 Prec=0.6312 | 🧠 Rec=0.6594
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.07it/s]
📉 Train Loss=0.4964 | Val Loss=0.5064 | 🎯 Dice=0.3049 | 📈 IoU=0.4630 | 🔍 Prec=0.6371 | 🧠 Rec=0.6306
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.97it/s]
📉 Train Loss=0.5026 | Val Loss=0.4836 | 🎯 Dice=0.3203 | 📈 IoU=0.5384 | 🔍 Prec=0.7201 | 🧠 Rec=0.6823
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.08s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.30it/s]
📉 Train Loss=0.5004 | Val Loss=0.4981 | 🎯 Dice=0.3126 | 📈 IoU=0.4839 | 🔍 Prec=0.6442 | 🧠 Rec=0.6623
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.28it/s]
📉 Train Loss=0.4877 | Val Loss=0.4879 | 🎯 Dice=0.3261 | 📈 IoU=0.5089 | 🔍 Prec=0.6609 | 🧠 Rec=0.6902
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.58it/s]
📉 Train Loss=0.4980 | Val Loss=0.4769 | 🎯 Dice=0.3412 | 📈 IoU=0.5438 | 🔍 Prec=0.6799 | 🧠 Rec=0.7327
✅ Saved new best model.

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.85it/s]
📉 Train Loss=0.4932 | Val Loss=0.4867 | 🎯 Dice=0.3207 | 📈 IoU=0.5128 | 🔍 Prec=0.6918 | 🧠 Rec=0.6661
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.40it/s]
📉 Train Loss=0.4910 | Val Loss=0.4846 | 🎯 Dice=0.3294 | 📈 IoU=0.5168 | 🔍 Prec=0.6637 | 🧠 Rec=0.7015
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.45it/s]
📉 Train Loss=0.4907 | Val Loss=0.4829 | 🎯 Dice=0.3304 | 📈 IoU=0.5205 | 🔍 Prec=0.6713 | 🧠 Rec=0.7002
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.48it/s]
📉 Train Loss=0.4883 | Val Loss=0.4911 | 🎯 Dice=0.3193 | 📈 IoU=0.4951 | 🔍 Prec=0.6710 | 🧠 Rec=0.6557
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.69it/s]
📉 Train Loss=0.4877 | Val Loss=0.5126 | 🎯 Dice=0.3014 | 📈 IoU=0.4430 | 🔍 Prec=0.6149 | 🧠 Rec=0.6148
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.41it/s]
📉 Train Loss=0.4971 | Val Loss=0.4898 | 🎯 Dice=0.3360 | 📈 IoU=0.4972 | 🔍 Prec=0.6187 | 🧠 Rec=0.7186
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.92it/s]
📉 Train Loss=0.4876 | Val Loss=0.4832 | 🎯 Dice=0.3320 | 📈 IoU=0.5123 | 🔍 Prec=0.6658 | 🧠 Rec=0.6910
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.08s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.64it/s]
📉 Train Loss=0.4930 | Val Loss=0.4852 | 🎯 Dice=0.3294 | 📈 IoU=0.5061 | 🔍 Prec=0.6504 | 🧠 Rec=0.6970
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.76it/s]
[I 2025-07-17 17:21:34,536] Trial 11 finished with value: 0.5437671542167664 and parameters: {'batch_size': 1, 'lr': 0.0004718570179612469, 'weight_decay': 0.0002260150612680787, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.4829 | Val Loss=0.4777 | 🎯 Dice=0.3421 | 📈 IoU=0.5281 | 🔍 Prec=0.6685 | 🧠 Rec=0.7169
⚠️ No improvement. wait_counter=9/12

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  7.77it/s]
📉 Train Loss=0.6640 | Val Loss=0.6132 | 🎯 Dice=0.2092 | 📈 IoU=0.3642 | 🔍 Prec=0.5151 | 🧠 Rec=0.5565
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.94it/s]
📉 Train Loss=0.5979 | Val Loss=0.5625 | 🎯 Dice=0.2598 | 📈 IoU=0.4460 | 🔍 Prec=0.5688 | 🧠 Rec=0.6749
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.46it/s]
📉 Train Loss=0.5586 | Val Loss=0.5398 | 🎯 Dice=0.2700 | 📈 IoU=0.4636 | 🔍 Prec=0.6155 | 🧠 Rec=0.6543
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.12it/s]
📉 Train Loss=0.5427 | Val Loss=0.5234 | 🎯 Dice=0.2907 | 📈 IoU=0.4781 | 🔍 Prec=0.6252 | 🧠 Rec=0.6714
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.92it/s]
📉 Train Loss=0.5309 | Val Loss=0.5152 | 🎯 Dice=0.2933 | 📈 IoU=0.4774 | 🔍 Prec=0.6389 | 🧠 Rec=0.6552
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.04it/s]
📉 Train Loss=0.5166 | Val Loss=0.5040 | 🎯 Dice=0.3004 | 📈 IoU=0.4952 | 🔍 Prec=0.6762 | 🧠 Rec=0.6505
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.68it/s]
📉 Train Loss=0.5109 | Val Loss=0.4950 | 🎯 Dice=0.3143 | 📈 IoU=0.5233 | 🔍 Prec=0.6670 | 🧠 Rec=0.7098
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.67it/s]
📉 Train Loss=0.5142 | Val Loss=0.4997 | 🎯 Dice=0.2998 | 📈 IoU=0.4918 | 🔍 Prec=0.7078 | 🧠 Rec=0.6184
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.80it/s]
📉 Train Loss=0.5063 | Val Loss=0.4951 | 🎯 Dice=0.3068 | 📈 IoU=0.5094 | 🔍 Prec=0.7315 | 🧠 Rec=0.6281
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.80it/s]
📉 Train Loss=0.5008 | Val Loss=0.4964 | 🎯 Dice=0.3076 | 📈 IoU=0.4920 | 🔍 Prec=0.6965 | 🧠 Rec=0.6274
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.87it/s]
📉 Train Loss=0.5012 | Val Loss=0.4849 | 🎯 Dice=0.3320 | 📈 IoU=0.5240 | 🔍 Prec=0.6647 | 🧠 Rec=0.7140
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.00s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.77it/s]
📉 Train Loss=0.4986 | Val Loss=0.5134 | 🎯 Dice=0.2961 | 📈 IoU=0.4423 | 🔍 Prec=0.6185 | 🧠 Rec=0.6097
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:29<00:00,  1.00it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.41it/s]
📉 Train Loss=0.4997 | Val Loss=0.5052 | 🎯 Dice=0.3072 | 📈 IoU=0.4632 | 🔍 Prec=0.6345 | 🧠 Rec=0.6335
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  7.40it/s]
📉 Train Loss=0.4984 | Val Loss=0.5016 | 🎯 Dice=0.3076 | 📈 IoU=0.4702 | 🔍 Prec=0.6598 | 🧠 Rec=0.6223
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.94it/s]
📉 Train Loss=0.4932 | Val Loss=0.4886 | 🎯 Dice=0.3224 | 📈 IoU=0.5015 | 🔍 Prec=0.6686 | 🧠 Rec=0.6689
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.52it/s]
📉 Train Loss=0.4870 | Val Loss=0.4821 | 🎯 Dice=0.3358 | 📈 IoU=0.5212 | 🔍 Prec=0.6685 | 🧠 Rec=0.7045
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.89it/s]
📉 Train Loss=0.4809 | Val Loss=0.4753 | 🎯 Dice=0.3400 | 📈 IoU=0.5452 | 🔍 Prec=0.6941 | 🧠 Rec=0.7191
✅ Saved new best model.

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.09it/s]
📉 Train Loss=0.4943 | Val Loss=0.4790 | 🎯 Dice=0.3388 | 📈 IoU=0.5326 | 🔍 Prec=0.6648 | 🧠 Rec=0.7294
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:29<00:00,  1.00it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.44it/s]
📉 Train Loss=0.4988 | Val Loss=0.4792 | 🎯 Dice=0.3350 | 📈 IoU=0.5313 | 🔍 Prec=0.6854 | 🧠 Rec=0.7039
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.01s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  9.72it/s]
📉 Train Loss=0.4848 | Val Loss=0.4734 | 🎯 Dice=0.3356 | 📈 IoU=0.5494 | 🔍 Prec=0.7223 | 🧠 Rec=0.6979
✅ Saved new best model.

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.77it/s]
📉 Train Loss=0.4924 | Val Loss=0.4786 | 🎯 Dice=0.3374 | 📈 IoU=0.5326 | 🔍 Prec=0.6671 | 🧠 Rec=0.7269
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.73it/s]
📉 Train Loss=0.4837 | Val Loss=0.4684 | 🎯 Dice=0.3533 | 📈 IoU=0.5589 | 🔍 Prec=0.6767 | 🧠 Rec=0.7641
✅ Saved new best model.

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:29<00:00,  1.00it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.68it/s]
📉 Train Loss=0.4880 | Val Loss=0.4783 | 🎯 Dice=0.3328 | 📈 IoU=0.5309 | 🔍 Prec=0.7068 | 🧠 Rec=0.6825
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.00s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.33it/s]
📉 Train Loss=0.4942 | Val Loss=0.4774 | 🎯 Dice=0.3383 | 📈 IoU=0.5320 | 🔍 Prec=0.6791 | 🧠 Rec=0.7118
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.79it/s]
[I 2025-07-17 17:34:40,559] Trial 12 finished with value: 0.5735729694366455 and parameters: {'batch_size': 1, 'lr': 0.0006344540966574607, 'weight_decay': 0.00021780823953025143, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.4900 | Val Loss=0.4636 | 🎯 Dice=0.3585 | 📈 IoU=0.5736 | 🔍 Prec=0.6956 | 🧠 Rec=0.7672
✅ Saved new best model.

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.22it/s]
📉 Train Loss=0.6631 | Val Loss=0.6191 | 🎯 Dice=0.1858 | 📈 IoU=0.3162 | 🔍 Prec=0.5334 | 🧠 Rec=0.4380
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.37it/s]
📉 Train Loss=0.5902 | Val Loss=0.5582 | 🎯 Dice=0.2510 | 📈 IoU=0.4425 | 🔍 Prec=0.6107 | 🧠 Rec=0.6174
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.67it/s]
📉 Train Loss=0.5537 | Val Loss=0.5293 | 🎯 Dice=0.2655 | 📈 IoU=0.4850 | 🔍 Prec=0.7095 | 🧠 Rec=0.6064
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.69it/s]
📉 Train Loss=0.5366 | Val Loss=0.5273 | 🎯 Dice=0.2900 | 📈 IoU=0.4573 | 🔍 Prec=0.5929 | 🧠 Rec=0.6680
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.27it/s]
📉 Train Loss=0.5272 | Val Loss=0.5188 | 🎯 Dice=0.2887 | 📈 IoU=0.4559 | 🔍 Prec=0.6116 | 🧠 Rec=0.6423
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.93it/s]
📉 Train Loss=0.5164 | Val Loss=0.5068 | 🎯 Dice=0.3083 | 📈 IoU=0.4770 | 🔍 Prec=0.6169 | 🧠 Rec=0.6792
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.41it/s]
📉 Train Loss=0.5076 | Val Loss=0.4956 | 🎯 Dice=0.3113 | 📈 IoU=0.5033 | 🔍 Prec=0.6620 | 🧠 Rec=0.6785
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.87it/s]
📉 Train Loss=0.5067 | Val Loss=0.5037 | 🎯 Dice=0.2911 | 📈 IoU=0.4741 | 🔍 Prec=0.7155 | 🧠 Rec=0.5857
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.32it/s]
📉 Train Loss=0.5041 | Val Loss=0.4968 | 🎯 Dice=0.3045 | 📈 IoU=0.4946 | 🔍 Prec=0.7045 | 🧠 Rec=0.6254
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.49it/s]
📉 Train Loss=0.5046 | Val Loss=0.4880 | 🎯 Dice=0.3187 | 📈 IoU=0.5181 | 🔍 Prec=0.7041 | 🧠 Rec=0.6632
✅ Saved new best model.

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.21it/s]
📉 Train Loss=0.4977 | Val Loss=0.4930 | 🎯 Dice=0.3221 | 📈 IoU=0.4948 | 🔍 Prec=0.6508 | 🧠 Rec=0.6750
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.10s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.54it/s]
📉 Train Loss=0.4965 | Val Loss=0.4936 | 🎯 Dice=0.3146 | 📈 IoU=0.4943 | 🔍 Prec=0.6870 | 🧠 Rec=0.6394
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.76it/s]
📉 Train Loss=0.4969 | Val Loss=0.4761 | 🎯 Dice=0.3402 | 📈 IoU=0.5460 | 🔍 Prec=0.6863 | 🧠 Rec=0.7287
✅ Saved new best model.

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.22it/s]
📉 Train Loss=0.5004 | Val Loss=0.4834 | 🎯 Dice=0.3276 | 📈 IoU=0.5220 | 🔍 Prec=0.6867 | 🧠 Rec=0.6867
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.09s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.18it/s]
📉 Train Loss=0.4966 | Val Loss=0.4828 | 🎯 Dice=0.3288 | 📈 IoU=0.5172 | 🔍 Prec=0.6788 | 🧠 Rec=0.6858
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.77it/s]
📉 Train Loss=0.5036 | Val Loss=0.4979 | 🎯 Dice=0.3184 | 📈 IoU=0.4769 | 🔍 Prec=0.6333 | 🧠 Rec=0.6605
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.27it/s]
📉 Train Loss=0.4871 | Val Loss=0.4848 | 🎯 Dice=0.3310 | 📈 IoU=0.5118 | 🔍 Prec=0.6547 | 🧠 Rec=0.7023
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.46it/s]
📉 Train Loss=0.4933 | Val Loss=0.5034 | 🎯 Dice=0.3132 | 📈 IoU=0.4640 | 🔍 Prec=0.6170 | 🧠 Rec=0.6529
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.03it/s]
📉 Train Loss=0.4949 | Val Loss=0.4767 | 🎯 Dice=0.3370 | 📈 IoU=0.5394 | 🔍 Prec=0.6922 | 🧠 Rec=0.7111
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.07it/s]
📉 Train Loss=0.4875 | Val Loss=0.4756 | 🎯 Dice=0.3428 | 📈 IoU=0.5359 | 🔍 Prec=0.6793 | 🧠 Rec=0.7188
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.98it/s]
📉 Train Loss=0.4903 | Val Loss=0.4763 | 🎯 Dice=0.3358 | 📈 IoU=0.5353 | 🔍 Prec=0.7105 | 🧠 Rec=0.6860
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.79it/s]
📉 Train Loss=0.4911 | Val Loss=0.4870 | 🎯 Dice=0.3348 | 📈 IoU=0.5019 | 🔍 Prec=0.6407 | 🧠 Rec=0.7001
⚠️ No improvement. wait_counter=9/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.09s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.34it/s]
📉 Train Loss=0.4838 | Val Loss=0.4773 | 🎯 Dice=0.3380 | 📈 IoU=0.5271 | 🔍 Prec=0.6866 | 🧠 Rec=0.6955
⚠️ No improvement. wait_counter=10/12

🔁 Epoch 24/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.02s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.10it/s]
📉 Train Loss=0.4765 | Val Loss=0.4677 | 🎯 Dice=0.3489 | 📈 IoU=0.5602 | 🔍 Prec=0.7027 | 🧠 Rec=0.7356
✅ Saved new best model.

🔁 Epoch 25/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.92it/s]
[I 2025-07-17 17:47:59,891] Trial 13 finished with value: 0.5602367401123047 and parameters: {'batch_size': 1, 'lr': 0.0007457509672133729, 'weight_decay': 5.5588012145470736e-05, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.4875 | Val Loss=0.4872 | 🎯 Dice=0.3347 | 📈 IoU=0.5017 | 🔍 Prec=0.6443 | 🧠 Rec=0.6954
⚠️ No improvement. wait_counter=1/12

➤ Starting training: model=unet, backbone=None, bs=1, img=256

🔁 Epoch 1/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.08s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.75it/s]
📉 Train Loss=0.6548 | Val Loss=0.6270 | 🎯 Dice=0.1259 | 📈 IoU=0.0874 | 🔍 Prec=0.5086 | 🧠 Rec=0.0948
✅ Saved new best model.

🔁 Epoch 2/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.11it/s]
📉 Train Loss=0.5899 | Val Loss=0.5419 | 🎯 Dice=0.2619 | 📈 IoU=0.3901 | 🔍 Prec=0.5654 | 🧠 Rec=0.5580
✅ Saved new best model.

🔁 Epoch 3/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.54it/s]
📉 Train Loss=0.5470 | Val Loss=0.5302 | 🎯 Dice=0.2595 | 📈 IoU=0.4088 | 🔍 Prec=0.6879 | 🧠 Rec=0.5087
✅ Saved new best model.

🔁 Epoch 4/25
Unfreezing backbone weights at epoch 4
Training: 100%|██████████| 30/30 [00:32<00:00,  1.08s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.92it/s]
📉 Train Loss=0.5346 | Val Loss=0.5107 | 🎯 Dice=0.2878 | 📈 IoU=0.4550 | 🔍 Prec=0.7093 | 🧠 Rec=0.5610
✅ Saved new best model.

🔁 Epoch 5/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.62it/s]
📉 Train Loss=0.5181 | Val Loss=0.5114 | 🎯 Dice=0.2954 | 📈 IoU=0.4564 | 🔍 Prec=0.6699 | 🧠 Rec=0.5902
✅ Saved new best model.

🔁 Epoch 6/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.12it/s]
📉 Train Loss=0.5136 | Val Loss=0.5071 | 🎯 Dice=0.3097 | 📈 IoU=0.4649 | 🔍 Prec=0.6215 | 🧠 Rec=0.6495
✅ Saved new best model.

🔁 Epoch 7/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.13it/s]
📉 Train Loss=0.5127 | Val Loss=0.5102 | 🎯 Dice=0.3092 | 📈 IoU=0.4698 | 🔍 Prec=0.6044 | 🧠 Rec=0.6792
✅ Saved new best model.

🔁 Epoch 8/25
Training: 100%|██████████| 30/30 [00:30<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.03it/s]
📉 Train Loss=0.5157 | Val Loss=0.5105 | 🎯 Dice=0.2855 | 📈 IoU=0.4559 | 🔍 Prec=0.7484 | 🧠 Rec=0.5394
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 9/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.07it/s]
📉 Train Loss=0.5148 | Val Loss=0.5053 | 🎯 Dice=0.2932 | 📈 IoU=0.4709 | 🔍 Prec=0.7309 | 🧠 Rec=0.5709
✅ Saved new best model.

🔁 Epoch 10/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.05s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.73it/s]
📉 Train Loss=0.5067 | Val Loss=0.5115 | 🎯 Dice=0.2991 | 📈 IoU=0.4506 | 🔍 Prec=0.6507 | 🧠 Rec=0.5955
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 11/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.03s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.39it/s]
📉 Train Loss=0.5069 | Val Loss=0.4883 | 🎯 Dice=0.3233 | 📈 IoU=0.5224 | 🔍 Prec=0.7109 | 🧠 Rec=0.6645
✅ Saved new best model.

🔁 Epoch 12/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.09s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.55it/s]
📉 Train Loss=0.5139 | Val Loss=0.5200 | 🎯 Dice=0.2892 | 📈 IoU=0.4287 | 🔍 Prec=0.6267 | 🧠 Rec=0.5769
⚠️ No improvement. wait_counter=1/12

🔁 Epoch 13/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.06s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.20it/s]
📉 Train Loss=0.5024 | Val Loss=0.4959 | 🎯 Dice=0.3211 | 📈 IoU=0.4949 | 🔍 Prec=0.6315 | 🧠 Rec=0.6970
⚠️ No improvement. wait_counter=2/12

🔁 Epoch 14/25
Training: 100%|██████████| 30/30 [00:31<00:00,  1.04s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.96it/s]
📉 Train Loss=0.5062 | Val Loss=0.5076 | 🎯 Dice=0.3084 | 📈 IoU=0.4574 | 🔍 Prec=0.6216 | 🧠 Rec=0.6349
⚠️ No improvement. wait_counter=3/12

🔁 Epoch 15/25
Training: 100%|██████████| 30/30 [00:32<00:00,  1.07s/it]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.85it/s]
📉 Train Loss=0.4991 | Val Loss=0.4984 | 🎯 Dice=0.3172 | 📈 IoU=0.4792 | 🔍 Prec=0.6272 | 🧠 Rec=0.6711
⚠️ No improvement. wait_counter=4/12

🔁 Epoch 16/25
Training: 100%|██████████| 30/30 [00:28<00:00,  1.04it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.15it/s]
📉 Train Loss=0.5055 | Val Loss=0.4927 | 🎯 Dice=0.3202 | 📈 IoU=0.4997 | 🔍 Prec=0.6540 | 🧠 Rec=0.6804
⚠️ No improvement. wait_counter=5/12

🔁 Epoch 17/25
Training: 100%|██████████| 30/30 [00:26<00:00,  1.13it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.87it/s]
📉 Train Loss=0.5079 | Val Loss=0.4948 | 🎯 Dice=0.3114 | 📈 IoU=0.4967 | 🔍 Prec=0.6781 | 🧠 Rec=0.6510
⚠️ No improvement. wait_counter=6/12

🔁 Epoch 18/25
Training: 100%|██████████| 30/30 [00:25<00:00,  1.19it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.66it/s]
📉 Train Loss=0.4907 | Val Loss=0.4884 | 🎯 Dice=0.3293 | 📈 IoU=0.5092 | 🔍 Prec=0.6603 | 🧠 Rec=0.6912
⚠️ No improvement. wait_counter=7/12

🔁 Epoch 19/25
Training: 100%|██████████| 30/30 [00:24<00:00,  1.20it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.64it/s]
📉 Train Loss=0.4912 | Val Loss=0.5056 | 🎯 Dice=0.3091 | 📈 IoU=0.4598 | 🔍 Prec=0.6276 | 🧠 Rec=0.6334
⚠️ No improvement. wait_counter=8/12

🔁 Epoch 20/25
Training: 100%|██████████| 30/30 [00:23<00:00,  1.27it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00,  8.79it/s]
📉 Train Loss=0.5046 | Val Loss=0.4973 | 🎯 Dice=0.3163 | 📈 IoU=0.4803 | 🔍 Prec=0.6561 | 🧠 Rec=0.6431
⚠️ No improvement. wait_counter=9/12

🔁 Epoch 21/25
Training: 100%|██████████| 30/30 [00:23<00:00,  1.26it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.52it/s]
📉 Train Loss=0.4933 | Val Loss=0.5076 | 🎯 Dice=0.3055 | 📈 IoU=0.4570 | 🔍 Prec=0.6514 | 🧠 Rec=0.6062
⚠️ No improvement. wait_counter=10/12

🔁 Epoch 22/25
Training: 100%|██████████| 30/30 [00:24<00:00,  1.23it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 11.07it/s]
📉 Train Loss=0.5002 | Val Loss=0.5057 | 🎯 Dice=0.3148 | 📈 IoU=0.4572 | 🔍 Prec=0.5957 | 🧠 Rec=0.6642
⚠️ No improvement. wait_counter=11/12

🔁 Epoch 23/25
Training: 100%|██████████| 30/30 [00:23<00:00,  1.25it/s]
Validating: 100%|██████████| 5/5 [00:00<00:00, 10.39it/s]
[I 2025-07-17 17:59:28,895] Trial 14 finished with value: 0.5224039435386658 and parameters: {'batch_size': 1, 'lr': 0.004797176509701105, 'weight_decay': 0.0005526662837063506, 'accumulation_steps': 1}. Best is trial 10 with value: 0.5786715745925903.
📉 Train Loss=0.5047 | Val Loss=0.4973 | 🎯 Dice=0.3149 | 📈 IoU=0.4814 | 🔍 Prec=0.6539 | 🧠 Rec=0.6468
⚠️ No improvement. wait_counter=12/12
"""

# Regex za epohu i metrike
epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
metrics_pattern = re.compile(
    r"Train Loss=([\d\.]+) \| Val Loss=([\d\.]+) \| 🎯 Dice=([\d\.]+) \| 📈 IoU=([\d\.]+) \| 🔍 Prec=([\d\.]+) \| 🧠 Rec=([\d\.]+)"
)

# Liste u koje čuvamo rezultate
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
        # Tražimo metrike u narednih 5 linija
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

# Formiraj DataFrame
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