import torch

from pathlib import Path
from omegaconf import DictConfig
from torch import nn

from models import custom


def build_model(cfg: DictConfig) -> nn.Module:
    if cfg.model.name == 'pixelencodernet':
        return custom.PixelEncoderNet(input_dim=cfg.model.in_channels, mlp1=[cfg.model.in_channels, 32, 64],
                                      output_dim=cfg.model.out_channels)


def save_model(network, cfg: DictConfig, model_name: str):
    save_file = Path(cfg.paths.output_path) / 'models' / f'{model_name}.pt'
    checkpoint = {
        'weights': network.state_dict(),
    }
    torch.save(checkpoint, save_file)


def load_model(cfg: DictConfig, model_name: str, device: torch.device):
    net = build_model(cfg)
    net.to(device)

    net_file = Path(cfg.paths.output_path) / 'models' / f'{model_name}.pt'

    checkpoint = torch.load(net_file, map_location=device)
    net.load_state_dict(checkpoint['weights'])

    return net