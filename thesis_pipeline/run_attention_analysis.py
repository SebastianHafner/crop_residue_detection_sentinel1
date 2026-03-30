"""
================================================================================
run_attention_analysis.py
================================================================================

WHAT IT DOES
------------
Loads the best trained PSENet model, runs inference on the test set with
attention extraction enabled, and produces four figures for the thesis:

  1. Mean temporal attention weight by day-of-year (DOY)
     → Shows which dates the model globally found most informative.
     → Expected peak: post-harvest window (DOY ~200–260) for winter wheat.

  2. Attention by DOY split by true class (residue vs no-residue)
     → Shows whether residue and no-residue fields have different temporal
       attention profiles.

  3. Per-field attention heatmap (top N fields sorted by P(residue))
     → Shows individual variation across fields.

  4. Attention entropy vs prediction confidence scatter
     → Tests whether confident predictions concentrate attention on fewer dates.

All figures saved as PDF to {output_path}/figures/.
A JSON summary of key attention statistics is saved for the thesis text.

USAGE
-----
  # Run after training — uses latest checkpoint automatically:
  cd /home/johan/Thesis
  python run_attention_analysis.py --experiment psenet_s1s2_E128_H4_20240901_120000

  # Or point directly at a checkpoint:
  python run_attention_analysis.py --checkpoint output/models/psenet_s1s2_E128_H4_20240901_120000.pt

OUTPUT
------
  {output_path}/figures/attn_by_doy.pdf
  {output_path}/figures/attn_by_class.pdf
  {output_path}/figures/attn_heatmap.pdf
  {output_path}/figures/attn_entropy_vs_confidence.pdf
  {output_path}/results/attention_summary.json
================================================================================
"""

import argparse
import json
from datetime import date
from pathlib import Path

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import entropy as scipy_entropy
from torch.utils import data as torch_data

from datasets.crop_dataset import CropDataset, collate_fn
from models.model_manager import load_model, build_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_PATH  = Path('/home/johan/Thesis/output')
FIGURES_PATH = OUTPUT_PATH / 'figures'
RESULTS_PATH = OUTPUT_PATH / 'results'

# Post-harvest window for winter wheat in Skåne (DOY)
HARVEST_START    = 172   # ~Jun 21
HARVEST_END      = 243   # ~Aug 31
POSTHARVEST_END  = 274   # ~Oct 1

# ---------------------------------------------------------------------------
# Inference with attention
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_with_attention(model, cfg: DictConfig, device: torch.device) -> list:
    """
    Run inference on the test set, extracting attention weights for each field.

    Returns:
        list of dicts, one per field, with keys:
          field_id, label, prediction, prob_residue, timestamps, attn_weights
    """
    ds = CropDataset(cfg, run_type='test', no_augmentations=True)
    loader = torch_data.DataLoader(
        ds,
        batch_size = cfg.trainer.batch_size,
        num_workers = 0,
        shuffle    = False,
        drop_last  = False,
        collate_fn = collate_fn,
    )

    model.eval()
    records = []

    for batch in loader:
        x    = batch['x'].to(device)
        doy  = batch['doy'].to(device)
        mask = batch['msk'].to(device)
        y    = batch['y']
        ids  = batch['id']

        logits, attn = model(x, doy, mask, return_attention=True)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        for i in range(len(ids)):
            records.append({
                'field_id':     ids[i],
                'label':        int(y[i].item()),
                'prediction':   int(preds[i].item()),
                'prob_residue': float(probs[i].item()),
                'doy_sequence': doy[i].cpu().numpy().tolist(),
                'attn_weights': attn[i].cpu().numpy().tolist(),
            })

    print(f'  Collected attention from {len(records)} test fields')
    return records


# ---------------------------------------------------------------------------
# Plot 1: Mean attention by DOY
# ---------------------------------------------------------------------------

def plot_mean_attention_by_doy(records: list, save_path: Path):
    doy_weights = {}
    for r in records:
        for d, w in zip(r['doy_sequence'], r['attn_weights']):
            doy = int(round(d))
            doy_weights.setdefault(doy, []).append(w)

    doys      = sorted(doy_weights.keys())
    mean_attn = [np.mean(doy_weights[d]) for d in doys]
    std_attn  = [np.std(doy_weights[d])  for d in doys]

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.bar(doys, mean_attn, width=3, color='#1D9E75', alpha=0.8, label='Mean attention')
    ax.fill_between(
        doys,
        np.array(mean_attn) - np.array(std_attn),
        np.array(mean_attn) + np.array(std_attn),
        alpha=0.2, color='#1D9E75',
    )

    ax.axvspan(HARVEST_START, HARVEST_END, alpha=0.08,
               color='#E24B4A', label='Harvest window (Jun–Aug)')
    ax.axvspan(HARVEST_END, POSTHARVEST_END, alpha=0.08,
               color='#EF9F27', label='Post-harvest (Sep–Oct)')

    ax.set_xlabel('Day of year', fontsize=12)
    ax.set_ylabel('Mean attention weight', fontsize=12)
    ax.set_title('Temporal attention — dates weighted by PSENet', fontsize=13)
    ax.set_xlim(1, 365)

    month_doys  = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax.set_xticks(month_doys)
    ax.set_xticklabels(month_names)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


# ---------------------------------------------------------------------------
# Plot 2: Attention by class
# ---------------------------------------------------------------------------

def plot_attention_by_class(records: list, save_path: Path):
    doy_weights = {0: {}, 1: {}}
    for r in records:
        lbl = r['label']
        if lbl not in (0, 1):
            continue
        for d, w in zip(r['doy_sequence'], r['attn_weights']):
            doy = int(round(d))
            doy_weights[lbl].setdefault(doy, []).append(w)

    colors     = {0: '#378ADD', 1: '#D85A30'}
    label_text = {0: 'No residue', 1: 'Residue'}

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for lbl, ax in zip([0, 1], axes):
        dws  = doy_weights[lbl]
        doys = sorted(dws.keys())
        mu   = [np.mean(dws[d]) for d in doys]
        sd   = [np.std(dws[d])  for d in doys]

        ax.bar(doys, mu, width=3, color=colors[lbl], alpha=0.75)
        ax.fill_between(doys,
            np.array(mu) - np.array(sd),
            np.array(mu) + np.array(sd),
            alpha=0.2, color=colors[lbl])

        ax.axvspan(HARVEST_START, HARVEST_END, alpha=0.06, color='#E24B4A')
        ax.axvspan(HARVEST_END,   POSTHARVEST_END, alpha=0.06, color='#EF9F27')
        ax.set_ylabel('Mean attention', fontsize=11)
        ax.set_title(label_text[lbl], fontsize=11)
        n = sum(1 for r in records if r['label'] == lbl)
        ax.text(0.98, 0.92, f'n={n}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

    month_doys  = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    axes[1].set_xticks(month_doys)
    axes[1].set_xticklabels(month_names)
    axes[1].set_xlabel('Day of year', fontsize=12)

    fig.suptitle('Temporal attention split by true residue class', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


# ---------------------------------------------------------------------------
# Plot 3: Per-field heatmap
# ---------------------------------------------------------------------------

def plot_field_attention_heatmap(records: list, save_path: Path, n_fields: int = 60):
    sorted_records = sorted(records, key=lambda r: -r['prob_residue'])[:n_fields]

    grid = np.zeros((len(sorted_records), 365))
    row_labels = []

    for i, r in enumerate(sorted_records):
        for d, w in zip(r['doy_sequence'], r['attn_weights']):
            doy = int(round(d)) - 1
            if 0 <= doy < 365:
                grid[i, doy] = w
        lbl = 'R' if r['label'] == 1 else 'N'
        row_labels.append(f"{r['field_id'][:8]}({lbl})")

    h = max(5, n_fields * 0.18)
    fig, ax = plt.subplots(figsize=(13, h))
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0)
    plt.colorbar(im, ax=ax, label='Attention weight', shrink=0.4)

    ax.axvspan(HARVEST_START - 1, HARVEST_END - 1, alpha=0.12, color='#E24B4A')
    ax.axvspan(HARVEST_END - 1, POSTHARVEST_END - 1, alpha=0.12, color='#EF9F27')

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    month_doys  = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax.set_xticks(month_doys)
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Day of year', fontsize=11)
    ax.set_title(f'Per-field attention (top {n_fields} by P(residue)) — R=residue, N=no residue',
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


# ---------------------------------------------------------------------------
# Plot 4: Attention entropy vs confidence
# ---------------------------------------------------------------------------

def plot_entropy_vs_confidence(records: list, save_path: Path):
    probs, entropies, labels = [], [], []
    for r in records:
        w = np.array(r['attn_weights'], dtype=np.float64)
        w = w / (w.sum() + 1e-8)
        probs.append(r['prob_residue'])
        entropies.append(float(scipy_entropy(w + 1e-8)))
        labels.append(r['label'])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        probs, entropies,
        c=labels, cmap='RdBu', alpha=0.55, s=18, vmin=0, vmax=1,
    )
    plt.colorbar(sc, ax=ax, label='True label (0=no residue, 1=residue)')
    ax.set_xlabel('P(residue) — model confidence', fontsize=12)
    ax.set_ylabel('Attention entropy (lower = more focused)', fontsize=12)
    ax.set_title('Confident predictions → focused temporal attention?', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {save_path}')


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_attention(records: list) -> dict:
    doy_weights = {}
    for r in records:
        for d, w in zip(r['doy_sequence'], r['attn_weights']):
            doy = int(round(d))
            doy_weights.setdefault(doy, []).append(w)

    mean_by_doy = {d: float(np.mean(ws)) for d, ws in doy_weights.items()}

    top5 = sorted(mean_by_doy.items(), key=lambda x: -x[1])[:5]
    top5_fmt = []
    for doy, w in top5:
        try:
            dt = date(2024, 1, 1).replace(
                month=1, day=1
            ).__class__.fromordinal(date(2024, 1, 1).toordinal() + doy - 1)
            top5_fmt.append({'doy': doy, 'date': dt.strftime('%b %d'), 'mean_weight': round(w, 5)})
        except Exception:
            top5_fmt.append({'doy': doy, 'date': 'n/a', 'mean_weight': round(w, 5)})

    ph_attn = sum(w for d, w in mean_by_doy.items() if HARVEST_END <= d <= POSTHARVEST_END)
    total   = sum(mean_by_doy.values()) + 1e-8

    n_correct = sum(1 for r in records if r['label'] == r['prediction'])
    f1_val = None
    try:
        from sklearn.metrics import f1_score
        f1_val = round(f1_score(
            [r['label'] for r in records],
            [r['prediction'] for r in records],
            average='binary', pos_label=1, zero_division=0,
        ), 4)
    except Exception:
        pass

    summary = {
        'n_test_fields':                 len(records),
        'test_accuracy':                 round(n_correct / len(records), 4),
        'test_f1_residue':               f1_val,
        'top5_attended_doys':            top5_fmt,
        'postharvest_attention_fraction': round(ph_attn / total, 4),
        'harvest_window_doy':            [HARVEST_START, HARVEST_END],
        'postharvest_window_doy':        [HARVEST_END, POSTHARVEST_END],
    }

    print('\n── Attention summary ──────────────────────────────')
    print(f'  Top 5 attended DOYs:')
    for item in top5_fmt:
        print(f'    DOY {item["doy"]:3d} ({item["date"]}) → {item["mean_weight"]:.5f}')
    print(f'  Post-harvest attention fraction '
          f'(DOY {HARVEST_END}–{POSTHARVEST_END}): '
          f'{summary["postharvest_attention_fraction"]:.1%}')

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=str, default=None,
                   help='Experiment name (model checkpoint stem)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Direct path to .pt checkpoint file')
    p.add_argument('--n_heatmap_fields', type=int, default=60,
                   help='Number of fields to show in heatmap')
    return p.parse_args()


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    args = parse_args()

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        saved_cfg = ckpt.get('cfg', cfg)
        model = build_model(saved_cfg)
        model.load_state_dict(ckpt['weights'])
        model.to(device)
        experiment_name = Path(args.checkpoint).stem
    elif args.experiment:
        model = load_model(cfg, args.experiment, device)
        experiment_name = args.experiment
    else:
        # Auto-find latest checkpoint
        model_dir = OUTPUT_PATH / 'models'
        pts = sorted(model_dir.glob('psenet_*.pt'), key=lambda p: p.stat().st_mtime)
        if not pts:
            raise FileNotFoundError(f'No checkpoints found in {model_dir}')
        latest = pts[-1]
        print(f'  Auto-loading latest checkpoint: {latest.name}')
        ckpt = torch.load(latest, map_location=device)
        saved_cfg = ckpt.get('cfg', cfg)
        model = build_model(saved_cfg)
        model.load_state_dict(ckpt['weights'])
        model.to(device)
        experiment_name = latest.stem

    print(f'\n── Running attention inference on test set ──')
    records = run_inference_with_attention(model, cfg, device)

    print('\n── Generating figures ──')
    plot_mean_attention_by_doy(
        records, FIGURES_PATH / f'{experiment_name}_attn_by_doy.pdf')
    plot_attention_by_class(
        records, FIGURES_PATH / f'{experiment_name}_attn_by_class.pdf')
    plot_field_attention_heatmap(
        records, FIGURES_PATH / f'{experiment_name}_attn_heatmap.pdf',
        n_fields=args.n_heatmap_fields)
    plot_entropy_vs_confidence(
        records, FIGURES_PATH / f'{experiment_name}_attn_entropy.pdf')

    summary = summarise_attention(records)

    summary_path = RESULTS_PATH / f'{experiment_name}_attention_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n✓ Attention summary saved → {summary_path}')


if __name__ == '__main__':
    main()
