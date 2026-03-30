"""
================================================================================
datasets/crop_dataset.py
================================================================================

WHAT IT DOES
------------
PyTorch Dataset for the crop residue time series pipeline.  Each sample is a
single agricultural field represented as a PIXEL SET — an unordered collection
of pixels within the field boundary — observed across T acquisition dates.

Input tensor shape going into the model: (T, C, P)
  T = number of time steps (acquisition dates)
  C = number of input channels (12: 6 S1 + 6 S2)
  P = fixed pixel set size (cfg.dataloader.n_pixels, default 64)

Mask tensor shape: (P,)  — 1 = valid pixel, 0 = padded pixel

Labels: integer 0 (no residue) or 1 (residue), from samples.json

The dataset also returns the day-of-year (DOY) sequence for each field,
derived from the metadata.json timestamps, used by the temporal attention
encoder for positional encoding.

S1 stacked TIF band order (written by s1_field_extraction.py):
  Band 1: VV          — gamma0 backscatter (dB), no normalisation
  Band 2: VH          — gamma0 backscatter (dB), no normalisation
  Band 3: Alpha       — 0-1 on disk (SNAP 0-90deg, normalised /90
                        by run_haalpha_timeseries.py)
  Band 4: Anisotropy  — 0-1 on disk (SNAP native, no change)
  Band 5: Entropy     — 0-1 on disk (SNAP native, no change)
  Band 6: DpRVI       — 0-1 on disk (SNAP native, no change)

When concatenated with S2 [B2,B3,B4,B8,B11,B12] the full 12-band input
vector to PSENet is:
  [VV, VH, Alpha, Anisotropy, Entropy, DpRVI, B2, B3, B4, B8, B11, B12]

USAGE
-----
  from datasets.crop_dataset import CropDataset
  ds = CropDataset(cfg, run_type='train')
  sample = ds[0]
  # sample keys: 'x' (T,C,P), 'doy' (T,), 'msk' (P,), 'y' (scalar), 'id' (str)

================================================================================
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Pixel-set time series dataset for crop residue classification.

    Loads per-field GeoTIFFs for each acquisition date, extracts pixels
    within the field mask, and applies pixel-set sampling to a fixed size.
    Returns the multi-sensor feature stack (S1 + S2), DOY sequence, and label.
    """

    def __init__(self, cfg: DictConfig, run_type: str, no_augmentations: bool = False):
        """
        Args:
            cfg:               Hydra DictConfig from config.yaml
            run_type:          'train', 'val', or 'test'
            no_augmentations:  If True, skip data augmentation (used for val/test)
        """
        super().__init__()
        self.cfg          = cfg
        self.run_type     = run_type
        self.dataset_path = Path(cfg.paths.dataset_path)
        self.modality     = cfg.dataloader.modality   # 's1s2' | 's1' | 's2'
        self.target       = cfg.dataloader.target     # 'residue_label'
        self.n_pixels     = cfg.dataloader.n_pixels   # fixed set size

        # Load samples list from JSON
        with open(cfg.paths.samples_file) as f:
            all_samples = json.load(f)

        set_id = {'train': 0, 'val': 1, 'test': 2}[run_type]
        self.samples = [s for s in all_samples if s['set'] == set_id]

        self.all_classes = sorted(set(s[self.target] for s in self.samples))
        assert len(self.all_classes) == cfg.model.out_channels, (
            f'Number of classes in dataset ({len(self.all_classes)}) does not '
            f'match cfg.model.out_channels ({cfg.model.out_channels})'
        )

        self.transform = self._build_transform(no_augmentations)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample    = self.samples[index]
        field_id  = str(sample['field_id'])
        field_dir = self.dataset_path / 'data' / field_id

        # Load metadata -> timestamps -> DOY sequence
        # Key is lowercase 'timestamps' as written by s1_field_extraction.py
        with open(field_dir / 'metadata.json') as f:
            meta = json.load(f)
        timestamps = meta['timestamps']   # list of ISO date strings e.g. '2024-06-07'

        doy = torch.tensor(
            [datetime.fromisoformat(ts).timetuple().tm_yday for ts in timestamps],
            dtype=torch.float32
        )  # (T,)

        # Load pixel arrays for each timestep -> (T, N_raw, C)
        x_raw, mask_raw = self._load_pixel_stack(field_id, timestamps)

        # Apply transforms: pixel-set sampling, augmentation, numpy->torch
        x, mask = self.transform((x_raw, mask_raw))
        # x: (T, C, P)   mask: (P,)

        y = torch.tensor(sample[self.target], dtype=torch.long)

        return {
            'x':   x.float(),      # (T, C, P)
            'doy': doy,            # (T,)
            'msk': mask.float(),   # (P,)
            'y':   y,              # scalar long
            'id':  field_id,
        }

    def __str__(self) -> str:
        n_res    = sum(s[self.target] for s in self.samples)
        n_no_res = len(self.samples) - n_res
        return (f'CropDataset [{self.run_type}] | '
                f'{len(self.samples)} fields | '
                f'residue={n_res} no_residue={n_no_res} | '
                f'modality={self.modality}')

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_pixel_stack(self, field_id: str, timestamps: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the multi-sensor feature stack for all timesteps.

        Returns:
            x_stack: (T, N_pixels, C)  — raw pixel values, all field pixels
            mask:    (N_pixels,)        — all True (real pixels, not padded yet)
        """
        field_dir = self.dataset_path / 'data' / field_id
        arrays    = []

        for ts in timestamps:
            bands = []

            if self.modality in ('s1s2', 's1'):
                s1 = self._load_s1(field_dir, ts, field_id)   # (N, 6)
                bands.append(s1)

            if self.modality in ('s1s2', 's2'):
                s2 = self._load_s2(field_dir, ts)              # (N, 6)
                bands.append(s2)

            x_t = np.concatenate(bands, axis=1)   # (N, C)
            arrays.append(x_t)

        x_stack = np.stack(arrays, axis=0)         # (T, N, C)
        mask    = np.ones(x_stack.shape[1], dtype=bool)
        return x_stack, mask

    def _load_s1(self, field_dir: Path, timestamp: str, field_id: str) -> np.ndarray:
        """
        Load the 6-band stacked Sentinel-1 TIF for one timestep.

        File written by s1_field_extraction.py:
          s1_{YYYYMMDD}_{field_id}.tif

        Band order (0-indexed in numpy, 1-indexed in rasterio):
          0: VV          — dB, no normalisation
          1: VH          — dB, no normalisation
          2: Alpha       — 0-1 (SNAP 0-90deg, normalised /90 on disk)
          3: Anisotropy  — 0-1 (SNAP native, no change needed)
          4: Entropy     — 0-1 (SNAP native, no change needed)
          5: DpRVI       — 0-1 (SNAP native, no change needed)

        All bands loaded as-is — no rescaling needed here.

        Returns: (N_pixels, 6)
        """
        # timestamps are ISO format '2024-06-07' -> convert to YYYYMMDD for filename
        date_str = timestamp.replace('-', '')
        tif_path = field_dir / f's1_{date_str}_{field_id}.tif'

        img = tifffile.imread(tif_path)   # (6, H, W) or (H, W, 6)

        # Ensure (bands, H, W)
        if img.ndim == 3 and img.shape[-1] == 6:
            img = img.transpose(2, 0, 1)

        assert img.shape[0] == 6, (
            f'Expected 6 bands in {tif_path.name}, got {img.shape[0]}. '
            f'Re-run s1_field_extraction.py.'
        )

        # Extract pixels within field boundary
        mask   = self._load_field_mask(field_dir)
        pixels = img[:, mask].T   # (N, 6)

        # All bands already correctly scaled on disk — no rescaling needed
        return pixels.astype(np.float32)

    def _load_s2(self, field_dir: Path, timestamp: str) -> np.ndarray:
        """
        Load Sentinel-2 bands for one timestep.

        Expected file: s2_{timestamp}.tif with bands:
          [B2, B3, B4, B8, B11, B12]  (raw int16 surface reflectance)

        Normalised here by dividing by cfg.dataloader.s2_normalization (10000).

        Returns: (N_pixels, 6)
        """
        tif_path = field_dir / f's2_{timestamp}.tif'
        img = tifffile.imread(tif_path)   # (6, H, W) or (H, W, 6)

        if img.ndim == 3 and img.shape[-1] == 6:
            img = img.transpose(2, 0, 1)

        mask   = self._load_field_mask(field_dir)
        pixels = img[:, mask].T   # (N, 6)

        # Normalise int16 reflectance to 0-1
        norm   = self.cfg.dataloader.s2_normalization
        pixels = np.clip(pixels / norm, 0.0, 1.0)

        return pixels.astype(np.float32)

    def _load_field_mask(self, field_dir: Path) -> np.ndarray:
        """Load the boolean field mask (True = inside field boundary)."""
        mask_path = field_dir / f'mask_{field_dir.name}.tif'
        mask = tifffile.imread(mask_path)
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------

    def _build_transform(self, no_augmentations: bool) -> transforms.Compose:
        tfms = [PixelSetSampler(self.n_pixels)]

        if not no_augmentations:
            aug_cfg = self.cfg.dataloader.augmentations
            if aug_cfg.color_shift:
                tfms.append(ColorShift())
            if aug_cfg.gamma_correction:
                tfms.append(GammaCorrection())

        tfms.append(Numpy2Torch())
        return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class PixelSetSampler:
    """
    Subsample or oversample pixels to a fixed set size.

    Large fields: random subsample without replacement.
    Small fields: fill remaining slots with a randomly chosen real pixel.
    Empty fields: return zero array with zero mask.
    """
    def __init__(self, set_size: int):
        self.set_size = set_size

    def __call__(self, inputs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, mask = inputs
        # x: (T, N, C),  mask: (N,) boolean

        T, N, C = x.shape
        P = self.set_size

        if N == 0:
            return np.zeros((T, P, C), dtype=np.float32), np.zeros(P, dtype=np.float32)

        if N >= P:
            idx = np.random.choice(N, size=P, replace=False)
            return x[:, idx, :], np.ones(P, dtype=np.float32)

        # N < P: pad with random real pixels
        x_out           = np.zeros((T, P, C), dtype=np.float32)
        x_out[:, :N, :] = x
        fill_idx        = np.random.choice(N, size=P - N, replace=True)
        x_out[:, N:, :] = x[:, fill_idx, :]
        return x_out, np.ones(P, dtype=np.float32)


class ColorShift:
    """Randomly scale each channel by a factor in [0.8, 1.2]."""
    def __init__(self, min_factor: float = 0.8, max_factor: float = 1.2):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, inputs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, mask = inputs
        factors = np.random.uniform(self.min_factor, self.max_factor, x.shape[-1])
        x = np.clip(x * factors[np.newaxis, np.newaxis, :], 0.0, 1.0).astype(np.float32)
        return x, mask


class GammaCorrection:
    """Randomly apply per-channel gamma correction."""
    def __init__(self, min_gamma: float = 0.7, max_gamma: float = 1.4):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, inputs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, mask = inputs
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, x.shape[-1])
        x = np.clip(np.power(x, gamma[np.newaxis, np.newaxis, :]), 0.0, 1.0).astype(np.float32)
        return x, mask


class Numpy2Torch:
    """
    Convert (T, N, C) numpy array to (T, C, N) torch tensor.
    Mask stays as (N,) tensor.
    """
    def __call__(self, inputs: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tensor, Tensor]:
        x, mask = inputs
        x_t    = torch.from_numpy(x).permute(0, 2, 1)   # (T, C, N)
        mask_t = torch.from_numpy(mask)
        return x_t, mask_t


# ---------------------------------------------------------------------------
# collate_fn — handles variable DOY lengths across a batch
# ---------------------------------------------------------------------------

def collate_fn(batch: list) -> dict:
    """
    Pads DOY sequences to the maximum T in the batch.
    All other tensors are already fixed-size.
    """
    max_T = max(item['doy'].shape[0] for item in batch)

    xs, doys, masks, ys, ids = [], [], [], [], []
    for item in batch:
        T   = item['doy'].shape[0]
        pad = max_T - T

        x = item['x']   # (T, C, P)
        if pad > 0:
            # Repeat last timestep to fill — model will attend these less
            x   = torch.cat([x, x[-1:].expand(pad, -1, -1)], dim=0)
            doy = torch.cat([item['doy'], item['doy'][-1:].expand(pad)], dim=0)
        else:
            doy = item['doy']

        xs.append(x)
        doys.append(doy)
        masks.append(item['msk'])
        ys.append(item['y'])
        ids.append(item['id'])

    return {
        'x':   torch.stack(xs),    # (B, T, C, P)
        'doy': torch.stack(doys),  # (B, T)
        'msk': torch.stack(masks), # (B, P)
        'y':   torch.stack(ys),    # (B,)
        'id':  ids,
    }