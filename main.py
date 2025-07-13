import yaml
import torch
import random
import numpy as np
from unitrain import train_model

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with open("configs/unet_exp1.yaml", "r") as f:
    config = yaml.safe_load(f)

if "seed" in config:
    set_seed(config["seed"])

train_model(data_dir=config["data_dir"], config=config)