"""
================================================================================
utils/loss_functions.py
================================================================================

WHAT IT DOES
------------
Returns the appropriate loss criterion based on the config.
Supports class-weighted CrossEntropyLoss to handle the imbalanced
residue / no-residue class distribution in the SBA survey data.

Class weights are read from cfg.trainer.class_weights:
  [1.0, 3.0]  → upweight the residue class by 3×

USAGE
-----
  from utils.loss_functions import get_criterion
  criterion = get_criterion(cfg, device)
  loss = criterion(logits, labels)   # logits: (B, 2), labels: (B,) long
================================================================================
"""

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_criterion(cfg: DictConfig, device: torch.device) -> nn.Module:
    """
    Build and return the loss criterion.

    Args:
        cfg:    Hydra DictConfig — reads trainer.loss and trainer.class_weights
        device: torch device to place weight tensor on

    Returns:
        nn.Module loss criterion
    """
    loss_type = cfg.trainer.loss

    if loss_type == 'CrossEntropy':
        weights = None
        if cfg.trainer.get('class_weights') is not None:
            weights = torch.tensor(
                list(cfg.trainer.class_weights), dtype=torch.float32
            ).to(device)
            print(f'  Loss: CrossEntropyLoss with class weights {list(cfg.trainer.class_weights)}')
        else:
            print('  Loss: CrossEntropyLoss (no class weights)')
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == 'FocalLoss':
        gamma  = cfg.trainer.get('focal_gamma', 2.0)
        print(f'  Loss: FocalLoss (gamma={gamma})')
        return FocalLoss(gamma=gamma)

    else:
        raise ValueError(f'Unknown loss type: {loss_type}. '
                         f'Supported: CrossEntropy, FocalLoss')


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced classification.
    Reduces the loss contribution from easy-to-classify examples,
    focusing training on the hard minority class (residue fields).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    """

    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt      = torch.exp(-ce_loss)
        fl      = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return fl.mean()
        elif self.reduction == 'sum':
            return fl.sum()
        return fl
