import torch
from segment_anything import sam_model_registry

def load_model(checkpoint_path, device):
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    return sam
