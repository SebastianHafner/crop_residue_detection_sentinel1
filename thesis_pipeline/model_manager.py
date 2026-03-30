"""
================================================================================
models/model_manager.py
================================================================================

WHAT IT DOES
------------
Central registry for building, saving, and loading models.  All model
construction goes through build_model() so train_network.py never imports
model classes directly — adding a new model only requires editing this file
and config.yaml.

Currently supported:
  cfg.model.name = 'psenet'   → full PSENet with temporal attention

USAGE
-----
  from models.model_manager import build_model, save_model, load_model

  model = build_model(cfg)
  save_model(model, cfg, experiment_name)
  model = load_model(cfg, experiment_name, device)
================================================================================
"""

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from models.psenet import PSENet


def build_model(cfg: DictConfig) -> nn.Module:
    """Construct and return the model specified in cfg.model.name."""
    name = cfg.model.name.lower()
    if name == 'psenet':
        return PSENet(cfg)
    else:
        raise ValueError(f'Unknown model: {cfg.model.name}. '
                         f'Supported: psenet')


def save_model(model: nn.Module, cfg: DictConfig, model_name: str) -> Path:
    """
    Save model weights + config to output_path/models/{model_name}.pt

    The config is saved alongside weights so load_model() can reconstruct
    the exact architecture without re-reading config.yaml.
    """
    save_dir = Path(cfg.paths.output_path) / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f'{model_name}.pt'

    torch.save({
        'weights': model.state_dict(),
        'cfg':     cfg,
    }, save_file)

    return save_file


def load_model(cfg: DictConfig, model_name: str, device: torch.device) -> nn.Module:
    """
    Load a saved model from output_path/models/{model_name}.pt

    Reconstructs the architecture from the saved cfg, loads weights,
    and moves to device.
    """
    net_file = Path(cfg.paths.output_path) / 'models' / f'{model_name}.pt'
    checkpoint = torch.load(net_file, map_location=device)

    # Use saved cfg to reconstruct architecture (handles config changes)
    saved_cfg = checkpoint.get('cfg', cfg)
    net = build_model(saved_cfg)
    net.load_state_dict(checkpoint['weights'])
    net.to(device)
    net.eval()
    return net
