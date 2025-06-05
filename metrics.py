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
    pred = torch.sigmoid(pred) > threshold
    if mask is not None:
        pred = pred * mask
        target = target * mask
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + eps) / (union + eps)
