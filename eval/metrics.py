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

        # normalize if necessary: check if max > 1 -indicating 0–255 input
        if mask.max() > 1:
            mask = mask / 255.0

        mask = (mask > 0.5)  # convert to boolean

        pred = pred & mask
        target = target & mask

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred & target).float().sum(dim=1)
    union = (pred | target).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean()

def precision_score(preds, targets, fov=None, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    if preds.shape != targets.shape:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
    if fov is not None:
        if fov.dim() == 3:
            fov = fov.unsqueeze(1)
        preds = preds * fov
        targets = targets * fov

    preds = preds.view(preds.size(0), -1)  # [B, N]
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1) # sample
    fp = (preds * (1 - targets)).sum(dim=1)

    precision = tp / (tp + fp + 1e-8)
    return precision.mean() #batch

def recall_score(preds, targets, fov=None, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    if preds.shape != targets.shape:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
    if fov is not None:
        if fov.dim() == 3:
            fov = fov.unsqueeze(1)
        preds = preds * fov
        targets = targets * fov

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    recall = tp / (tp + fn + 1e-8)
    return recall.mean()