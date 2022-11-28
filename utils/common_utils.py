import numpy as np
import random
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed: int, is_deterministic: bool = True):
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