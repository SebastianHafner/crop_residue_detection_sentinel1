"""
================================================================================
utils/evaluation.py
================================================================================

WHAT IT DOES
------------
Runs evaluation on val or test split and returns metrics used for:
  - Early stopping (val F1, primary metric)
  - WandB logging (loss, OA, F1, AUC)
  - Final thesis results table (full classification report)

Primary metric returned: F1 score for the residue class (label=1).
This is preferred over overall accuracy because the dataset is class-imbalanced —
a model predicting all no-residue would achieve high OA but zero F1.

USAGE
-----
  from utils.evaluation import run_evaluation
  metrics = run_evaluation(model, cfg, device, run_type='val',
                            epoch=1.0, step=100)
  val_f1 = metrics['f1']
================================================================================
"""

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils import data as torch_data

from datasets.crop_dataset import CropDataset, collate_fn
from utils.loss_functions import get_criterion


def run_evaluation(
    model,
    cfg:      DictConfig,
    device:   torch.device,
    run_type: str,           # 'val' or 'test'
    epoch:    float,
    step:     int,
    max_samples: int = None,
    verbose:  bool  = False,
) -> dict:
    """
    Evaluate the model on the given split.

    Args:
        model:       PSENet (or any model with forward(x, doy, mask))
        cfg:         Hydra DictConfig
        device:      torch device
        run_type:    'val' | 'test'
        epoch:       current training epoch (float, for logging)
        step:        current global step (for logging)
        max_samples: if set, evaluate on at most this many samples (for speed)
        verbose:     if True, print full classification report

    Returns:
        dict with keys: loss, oa, f1, f1_macro, auc, confusion_matrix
    """
    ds = CropDataset(cfg, run_type=run_type, no_augmentations=True)
    if max_samples is not None:
        indices = np.random.choice(len(ds), min(max_samples, len(ds)), replace=False)
        ds      = torch.utils.data.Subset(ds, indices)

    criterion = get_criterion(cfg, device)

    loader = torch_data.DataLoader(
        ds,
        batch_size  = cfg.trainer.batch_size,
        num_workers = 0,
        shuffle     = False,
        drop_last   = False,
        collate_fn  = collate_fn,
    )

    model.eval()
    loss_vals, all_preds, all_probs, all_labels = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x    = batch['x'].to(device)    # (B, T, C, P)
            doy  = batch['doy'].to(device)  # (B, T)
            mask = batch['msk'].to(device)  # (B, P)
            y    = batch['y'].to(device)    # (B,)

            logits = model(x, doy, mask)
            loss   = criterion(logits, y)
            probs  = torch.softmax(logits, dim=1)[:, 1]   # P(residue)
            preds  = logits.argmax(dim=1)

            loss_vals.extend([loss.item()] * len(y))
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    loss_mean = float(np.mean(loss_vals))
    oa        = float(accuracy_score(all_labels, all_preds))
    f1        = float(f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0))
    f1_macro  = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))

    # AUC requires both classes present
    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    metrics = {
        'loss':             loss_mean,
        'oa':               oa,
        'f1':               f1,
        'f1_macro':         f1_macro,
        'auc':              auc,
        'confusion_matrix': cm,
    }

    # WandB logging
    wandb.log({
        f'{run_type}/loss':     loss_mean,
        f'{run_type}/oa':       oa,
        f'{run_type}/f1':       f1,
        f'{run_type}/f1_macro': f1_macro,
        f'{run_type}/auc':      auc,
        'epoch': epoch,
        'step':  step,
    })

    print(f'[{run_type}] epoch={epoch:.2f} | '
          f'loss={loss_mean:.4f} | OA={oa:.3f} | '
          f'F1={f1:.3f} | F1_macro={f1_macro:.3f} | AUC={auc:.3f}')

    if verbose:
        print(classification_report(
            all_labels, all_preds,
            target_names=['no_residue', 'residue'],
            zero_division=0,
        ))
        print(f'Confusion matrix:\n  pred→  no_res  res')
        print(f'  no_res  {cm[0][0]:5d}  {cm[0][1]:4d}')
        print(f'  res     {cm[1][0]:5d}  {cm[1][1]:4d}')

    return metrics
