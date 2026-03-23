import torch
import torch.nn as nn
from typing import Tuple


def get_criterion(loss_type: str, class_weights: Tuple[float, ...] = None):
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f'Unknown loss {loss_type}')
    return criterion

