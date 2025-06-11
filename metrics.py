import torch
import torch.nn.functional as F

def dice_loss(pred, target, mask=None, smooth=1e-6):
    pred = torch.sigmoid(pred)
    if mask is not None:
        pred = pred * mask
        target = target * mask
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def iou_score(pred, target, mask=None, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).bool()
    target = (target > 0.5).bool()
    
    if mask is not None:
        mask = (mask > 0.5).bool()
        pred = pred & mask
        target = target & mask

    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + eps) / (union + eps)