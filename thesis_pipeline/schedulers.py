"""
================================================================================
utils/schedulers.py
================================================================================

WHAT IT DOES
------------
Returns a PyTorch learning rate scheduler based on cfg.trainer.lr_scheduler.

Supported schedulers:
  plateau  → ReduceLROnPlateau: reduces LR when val loss stops improving.
             Best default for this task — handles varying convergence speed.
  step     → StepLR: reduce by gamma every N epochs (epochs // 3)
  linear   → LambdaLR: linearly decay to zero over training
  cosine   → CosineAnnealingLR: smooth cosine decay
  none     → no scheduler (constant LR)

USAGE
-----
  from utils.schedulers import get_scheduler
  scheduler = get_scheduler(cfg, optimizer)

  # In training loop (for all except plateau):
  scheduler.step()

  # For plateau — call after computing val loss:
  scheduler.step(val_loss)
================================================================================
"""

from torch.optim import lr_scheduler, Optimizer
from omegaconf import DictConfig


def get_scheduler(cfg: DictConfig, optimizer: Optimizer):
    """
    Build and return a learning rate scheduler.

    Args:
        cfg:       Hydra DictConfig — reads trainer.lr_scheduler and trainer.epochs
        optimizer: the model optimizer

    Returns:
        scheduler object, or None if lr_scheduler == 'none'
    """
    name = cfg.trainer.lr_scheduler.lower()

    if name == 'none':
        print('  LR scheduler: none (constant LR)')
        return None

    elif name == 'plateau':
        # Reduce LR by 0.5 when val loss doesn't improve for 5 epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
            min_lr=1e-6, verbose=True,
        )
        print('  LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)')
        return scheduler

    elif name == 'step':
        step_size = max(1, cfg.trainer.epochs // 3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        print(f'  LR scheduler: StepLR (step_size={step_size}, gamma=0.1)')
        return scheduler

    elif name == 'linear':
        def lambda_rule(epoch):
            return max(0.0, 1.0 - epoch / float(cfg.trainer.epochs - 1))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        print('  LR scheduler: Linear decay to 0')
        return scheduler

    elif name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.trainer.epochs, eta_min=1e-6
        )
        print(f'  LR scheduler: CosineAnnealingLR (T_max={cfg.trainer.epochs})')
        return scheduler

    else:
        raise ValueError(f'Unknown scheduler: {cfg.trainer.lr_scheduler}. '
                         f'Supported: plateau, step, linear, cosine, none')
