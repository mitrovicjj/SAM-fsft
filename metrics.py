import torch
import torch.nn.functional as F

def dice_loss(pred, target, mask=None, smooth=1e-6):
    pred = torch.sigmoid(pred)
    
    if pred.shape != target.shape:
        if target.dim() == 3:
            target = target.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]
        pred = pred * mask
        target = target * mask

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1 - dice.mean()

def iou_score(pred, target, mask=None, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold)
    target = (target > 0.5)

    if pred.shape != target.shape:
        if target.dim() == 3:
            target = target.unsqueeze(1)

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        pred = pred & mask
        target = target & mask

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred & target).float().sum(dim=1)
    union = (pred | target).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean()

%load_ext tensorboard
%tensorboard --logdir runs
