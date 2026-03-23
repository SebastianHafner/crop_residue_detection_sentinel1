from pathlib import Path
from abc import abstractmethod
from typing import Tuple
from omegaconf import DictConfig

import torch
from torch import Tensor
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder

import numpy as np
import tifffile
import json


class AbstractCropDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset_path = Path(cfg.paths.dataset_path)
        self.samples_file = Path(cfg.paths.samples_file)
        self.modality = cfg.dataloader.modality
        self.target = cfg.dataloader.target
        self.load_pixelsets = cfg.dataloader.load_pixelsets
        self.name = f'{self.modality}'

    def _load_mask(self, field_id: str) -> np.ndarray:
        file_name = self.dataset_path / 'data' / field_id / f'mask_{field_id}.tif'
        mask = tifffile.imread(file_name)
        return mask.astype(np.bool)

    def _load_input(self, field_id: str, modality: str) -> np.ndarray:
        if modality == 'embedding':
            return self._load_pixelset_embedding(field_id) if self.load_pixelsets else self._load_embedding(field_id)
        elif modality == 's2':
            return self._load_s2(field_id)

    def _load_s2(self, field_id: str) -> np.ndarray:
        file_name = self.dataset_path / 'data' / field_id / f's2_{field_id}.tif'
        img = tifffile.imread(file_name)
        img = img[:, :, self.cfg.dataloader.band_indices]
        img = np.clip(img / self.cfg.dataloader.normalization_value, 0, 1)
        return img.astype(np.float32)

    def _load_pixelset_embedding(self, field_id: str) -> np.ndarray:
        file_name = self.dataset_path / 'data' / field_id / f'embeddings_pixelset_{field_id}.npy'
        pixelset = np.load(file_name)
        return pixelset

    def _load_embedding(self, field_id: str) -> np.ndarray:
        file_name = self.dataset_path / 'data' / field_id / f'embeddings_{field_id}.tif'
        img = tifffile.imread(file_name)
        return img.astype(np.float32)

    def _compose_transformations(self, no_augmentations: bool = False) -> transforms.Compose:
        transformations = []
        if self.cfg.dataloader.augmentations.pixel_set:
            transformations.append(PixelSet(self.cfg.dataloader.augmentations.n_pixels, self.load_pixelsets))
        if not no_augmentations:
            if self.cfg.dataloader.augmentations.random_flip:
                transformations.append(RandomFlip())
            if self.cfg.dataloader.augmentations.random_rotate:
                transformations.append(RandomRotate())
            if self.cfg.dataloader.augmentations.color_shift:
                transformations.append(ColorShift())
            if self.cfg.dataloader.augmentations.gamma_correction:
                transformations.append(GammaCorrection())
        transformations.append(Numpy2Torch())
        return transforms.Compose(transformations)


    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class CropDataset(AbstractCropDataset):

    def __init__(self, cfg: DictConfig, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations

        with open(self.samples_file) as f:
            self.all_samples = json.load(f)

        set_id = 0 if run_type == 'train' else 1 if run_type == 'val' else 2
        self.samples = [s for s in self.all_samples if s['set'] == set_id]

        self.all_classes = list(set([s[self.target] for s in self.samples]))
        self.le = LabelEncoder()
        self.le.fit(self.all_classes)

        self.transform = self._compose_transformations(no_augmentations)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        field_id = str(sample['field_id'])

        # x (C, H, W) or (C, P)
        x = self._load_input(field_id, self.modality)
        # mask (H, W) or (P)
        mask = np.ones((x.shape[0]), dtype=bool) if self.load_pixelsets else self._load_mask(field_id)

        # Add temporal dimension
        if len(x.shape) < 4:
            x = x[np.newaxis]
        if len(x.shape) == 4:
            x = x.transpose((0, 2, 3, 1))
        # x (T x H x W x C) - for pixelsets the input is (T x P x C)

        x, mask = self.transform((x, mask))
        y = self.le.transform([sample[self.target]])

        item = {
            'x': x.float(),
            'y': torch.tensor(y[0]).long(),
            'msk': mask.int(),
            'id': field_id,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


class PixelSet(object):
    def __init__(self, n_pixels: int, pixelset_input: bool = False):
        self.set_size = n_pixels
        self.pixelset_input = pixelset_input

    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        The input of the Pixel Set is a tuple of arrays:
          (Images-Time-Series, Field-Mask)
        Image-Time-Series : (Sequence length) x Height x Width x Channel
        Field-Mask : Height x Width
        """
        imgs, mask = input_arrays
        field_pixels = imgs[:, mask]
        # field_pixels (T x P x C), where P is the number of unmasked pixels
        ts_length, n_field_pixels, bands = field_pixels.shape

        # Subsample large fields
        if n_field_pixels > self.set_size:
            idx = np.random.choice(list(range(n_field_pixels)), size=self.set_size, replace=False)
            x = field_pixels[:, idx]
            mask = np.ones(self.set_size)

        # Oversample small fields
        elif n_field_pixels < self.set_size:

            # TODO: See if this is necessary (empty fields)
            if n_field_pixels == 0:
                x = np.zeros((ts_length, self.set_size, bands))
                mask = np.zeros(self.set_size)
            else:
                x = np.zeros((ts_length, self.set_size, bands))
                x[:, :n_field_pixels] = field_pixels
                # Instead of field_pixels[:, 0] as in original code (paper says random)
                repeated_pixel = field_pixels[:, np.random.randint(0, n_field_pixels) ]
                fill_pixels = np.stack([repeated_pixel for _ in range(n_field_pixels, self.set_size)], axis=-1)
                x[:, n_field_pixels:] = fill_pixels.transpose(0, 2, 1)
                mask = np.ones(self.set_size)
        else:
            x = field_pixels
            mask = np.ones(self.set_size)
        return x, mask


class Numpy2Torch(object):
    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[Tensor, Tensor]:
        x, mask = input_arrays
        x_tensor, mask_tensor = torch.from_numpy(x), torch.from_numpy(mask)
        if len(x_tensor.shape) == 3:  # (T x P x C) to (T x C x P)
            x_tensor = x_tensor.permute(0, 2, 1)
        if len(x_tensor.shape) == 4:  # (T x H x W x C) to (T x C x H x W)
            x_tensor = x_tensor.permute(0, 3, 1, 2)
        return x_tensor, mask_tensor


class RandomFlip(object):
    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = input_arrays
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)

        img = img.copy()
        mask = mask.copy()

        return img, mask


class RandomRotate(object):
    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = input_arrays
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return img, mask


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = input_arrays
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, mask


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, input_arrays: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = input_arrays
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img.shape[-1])
        img_gamma_corrected = np.clip(np.power(img, gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_gamma_corrected, mask
