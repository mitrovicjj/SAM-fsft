import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

def get_segformer_model(backbone: str,
                        num_labels: int,
                        freeze_backbone_epochs: int = 0):
    """
    Returns a SegFormer with a re‑initialized 1‑class head.
    
    Args
    ----
    backbone : str
        HuggingFace model ID, e.g. "nvidia/segformer-b0-finetuned-ade-512-512".
    num_labels : int
        Number of output channels (1 for retinal vessel mask).
    freeze_backbone_epochs : int
        If >0, the caller should train with encoder frozen for this many
        epochs, then un‑freeze and continue fine‑tuning.
    """
    # 1. Load pretrained checkpoint, letting HF replace the classifier tensor
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # 2. Re‑initialise the new classifier properly
    def _init_classifier(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.decode_head.classifier.apply(_init_classifier)

    # (Optional) also re‑init auxiliary head if you use it:
    if hasattr(model, "auxiliary_head"):
        model.auxiliary_head.classifier.apply(_init_classifier)

    # 3. Optionally freeze encoder so the new head can stabilise
    if freeze_backbone_epochs > 0:
        for p in model.segformer.parameters():
            p.requires_grad = False
        model.segformer.__dict__["_freezed_for"] = freeze_backbone_epochs  # stash flag

    return model