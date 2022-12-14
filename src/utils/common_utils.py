import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import yaml
from torch import nn


def setup_experiment_folder(outputs_dir: str):
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    outputs_dir = os.path.join(outputs_dir, now)
    checkpoint_path = os.path.join(outputs_dir, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    return outputs_dir, checkpoint_path


def setup_device(device: Optional[int] = None):
    if torch.cuda.is_available():
        device = f"cuda:{device}"
    else:
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except:
            device = "cpu"
    return torch.device(device)


def setup_random_seed(seed: int, is_deterministic: bool = True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_yaml(filepath: str):
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc
