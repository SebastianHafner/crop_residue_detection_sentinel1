"""
================================================================================
train_network.py
================================================================================

WHAT IT DOES
------------
Main training entry point for the crop residue PSENet pipeline.

Training loop:
  1. Load dataset (train + val splits from samples.json)
  2. Build PSENet model from config
  3. For each epoch:
       a. Forward pass on all training batches
       b. Backward pass + AdamW optimizer step
       c. Gradient clipping (prevents transformer instability)
       d. Log training loss to WandB every cfg.log_freq steps
       e. Run validation evaluation (F1 is primary metric)
       f. Step LR scheduler
       g. Save checkpoint if val F1 improved
       h. Early stopping if patience exceeded
  4. Load best checkpoint, run final test evaluation

All metrics are logged to WandB.  Checkpoints saved to:
  {output_path}/models/{experiment_name}.pt

USAGE
-----
  # Standard training run:
  cd /home/johan/Thesis
  python train_network.py

  # Override config values from command line (Hydra syntax):
  python train_network.py trainer.lr=5e-4 model.n_heads=8

  # Debug mode (disables WandB, uses 1 worker, small eval):
  python train_network.py debug=true

  # Hyperparameter sweep:
  python train_network.py -m model.embed_dim=64,128,256 trainer.lr=1e-3,5e-4

REQUIREMENTS
------------
  pip install torch torchvision hydra-core omegaconf wandb \
              scikit-learn geopandas tifffile numpy tqdm python-dotenv

OUTPUT
------
  {output_path}/models/{experiment}.pt       — best model checkpoint
  {output_path}/results/test_metrics.json    — final test metrics
  WandB run at https://wandb.ai/{entity}/{project}
================================================================================
"""

import datetime
import json
import random
import timeit
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data

import hydra

from datasets.crop_dataset import CropDataset, collate_fn
from models.model_manager import build_model, save_model, load_model
from utils.evaluation import run_evaluation
from utils.loss_functions import get_criterion
from utils.schedulers import get_scheduler


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='config', config_name='config')
def run_training(cfg: DictConfig) -> None:

    print('\n' + '='*70)
    print('  Crop Residue Detection — PSENet Training')
    print('='*70)
    print(OmegaConf.to_yaml(cfg))

    # ── Reproducibility ────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # ── Paths ──────────────────────────────────────────────────────────
    output_path = Path(cfg.paths.output_path)
    (output_path / 'models').mkdir(parents=True, exist_ok=True)
    (output_path / 'results').mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────
    train_ds = CropDataset(cfg, run_type='train')
    print(f'\n{train_ds}')

    assert len(train_ds.all_classes) == cfg.model.out_channels, (
        f'Dataset has {len(train_ds.all_classes)} classes but '
        f'cfg.model.out_channels={cfg.model.out_channels}')

    # ── Experiment name ────────────────────────────────────────────────
    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment = (f'{cfg.model.name}'
                  f'_{cfg.dataloader.modality}'
                  f'_E{cfg.model.embed_dim}'
                  f'_H{cfg.model.n_heads}'
                  f'_{timestamp}')
    print(f'\nExperiment: {experiment}')

    # ── WandB ──────────────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb.project,
        entity  = cfg.wandb.get('entity'),
        name    = experiment,
        config  = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode    = 'disabled' if cfg.debug else 'online',
    )

    # ── Model ──────────────────────────────────────────────────────────
    print('\n── Building model ──')
    model = build_model(cfg)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')

    # ── Loss, optimizer, scheduler ────────────────────────────────────
    criterion = get_criterion(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.trainer.lr,
        weight_decay = cfg.trainer.weight_decay,
    )
    scheduler = get_scheduler(cfg, optimizer)

    # ── DataLoader ─────────────────────────────────────────────────────
    dataloader = torch_data.DataLoader(
        train_ds,
        batch_size  = cfg.trainer.batch_size,
        num_workers = 0 if cfg.debug else cfg.dataloader.num_workers,
        shuffle     = cfg.dataloader.shuffle,
        drop_last   = True,
        pin_memory  = device.type == 'cuda',
        collate_fn  = collate_fn,
    )

    steps_per_epoch = len(dataloader)
    print(f'  Steps per epoch: {steps_per_epoch}')

    # ── Training state ─────────────────────────────────────────────────
    global_step    = 0
    epoch_float    = 0.0
    best_val_f1    = -1.0
    trigger_times  = 0
    stop_training  = False

    # Initial val baseline before any training
    print('\n── Initial validation ──')
    init_metrics = run_evaluation(model, cfg, device, 'val', 0.0, 0,
                                  cfg.trainer.max_eval_samples)
    best_val_f1  = init_metrics['f1']

    # ── Epoch loop ─────────────────────────────────────────────────────
    print(f'\n── Training ({cfg.trainer.epochs} epochs) ──')
    for epoch in range(1, cfg.trainer.epochs + 1):
        model.train()
        loss_set   = []
        start_time = timeit.default_timer()

        for i, batch in enumerate(dataloader):
            x    = batch['x'].to(device)    # (B, T, C, P)
            doy  = batch['doy'].to(device)  # (B, T)
            mask = batch['msk'].to(device)  # (B, P)
            y    = batch['y'].to(device)    # (B,)

            optimizer.zero_grad()

            logits = model(x, doy, mask)    # (B, n_classes)
            loss   = criterion(logits, y)

            loss.backward()

            # Gradient clipping — important for transformer stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.trainer.grad_clip
            )

            optimizer.step()

            loss_set.append(loss.item())
            global_step += 1
            epoch_float  = global_step / steps_per_epoch

            if global_step % cfg.log_freq == 0:
                elapsed = timeit.default_timer() - start_time
                wandb.log({
                    'train/loss': np.mean(loss_set),
                    'train/time': elapsed,
                    'epoch':      epoch_float,
                    'step':       global_step,
                })
                print(f'  step={global_step:5d} '
                      f'epoch={epoch_float:.2f} '
                      f'loss={np.mean(loss_set):.4f} '
                      f'({elapsed:.1f}s)')
                loss_set   = []
                start_time = timeit.default_timer()

        # ── End of epoch ───────────────────────────────────────────────
        assert abs(epoch - epoch_float) < 0.01, \
            f'Epoch counter mismatch: epoch={epoch} epoch_float={epoch_float:.3f}'

        # Validation
        val_metrics = run_evaluation(
            model, cfg, device, 'val',
            epoch_float, global_step,
            cfg.trainer.max_eval_samples,
        )
        val_f1 = val_metrics['f1']

        # LR scheduler step
        if scheduler is not None:
            if cfg.trainer.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({'train/lr': current_lr, 'epoch': epoch_float, 'step': global_step})

        # Checkpoint + early stopping
        print(f'  val F1={val_f1:.4f} (best={best_val_f1:.4f}) '
              f'patience={trigger_times}/{cfg.trainer.patience} '
              f'lr={current_lr:.2e}')

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            trigger_times = 0
            saved = save_model(model, cfg, experiment)
            wandb.log({'best_val_f1': best_val_f1, 'step': global_step})
            print(f'  ✓ New best val F1={best_val_f1:.4f} — saved to {saved}')
        else:
            trigger_times += 1
            if trigger_times >= cfg.trainer.patience:
                print(f'\n  Early stopping at epoch {epoch} '
                      f'(no improvement for {cfg.trainer.patience} epochs)')
                stop_training = True

        if stop_training:
            break

    # ── Final test evaluation ──────────────────────────────────────────
    print('\n── Loading best model for final test evaluation ──')
    best_model = load_model(cfg, experiment, device)

    test_metrics = run_evaluation(
        best_model, cfg, device, 'test',
        epoch_float, global_step, verbose=True,
    )

    # Save metrics to JSON for thesis reporting
    results_file = output_path / 'results' / f'{experiment}_test_metrics.json'
    with open(results_file, 'w') as f:
        json.dump({
            'experiment':   experiment,
            'best_val_f1':  best_val_f1,
            'test_metrics': test_metrics,
            'cfg':          OmegaConf.to_container(cfg, resolve=True),
        }, f, indent=2)

    print(f'\n✓ Test metrics saved → {results_file}')
    print(f'✓ Best model      → {output_path}/models/{experiment}.pt')

    wandb.log({
        'test/final_f1':  test_metrics['f1'],
        'test/final_oa':  test_metrics['oa'],
        'test/final_auc': test_metrics['auc'],
    })
    wandb.finish()

    print('\n' + '='*70)
    print(f'  Training complete.')
    print(f'  Best val F1:  {best_val_f1:.4f}')
    print(f'  Test F1:      {test_metrics["f1"]:.4f}')
    print(f'  Test OA:      {test_metrics["oa"]:.4f}')
    print(f'  Test AUC:     {test_metrics["auc"]:.4f}')
    print('='*70)


if __name__ == '__main__':
    run_training()
