import torch

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