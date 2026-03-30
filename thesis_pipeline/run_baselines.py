"""
================================================================================
run_baselines.py
================================================================================

WHAT IT DOES
------------
Trains and evaluates three classical ML baselines on the same train/test split
as PSENet, for the thesis comparison table:
  - Random Forest
  - Gradient Boosting
  - SVM (RBF kernel)

Because these models cannot consume (T, C, P) pixel-set tensors, each field
is represented as a FLAT FEATURE VECTOR of hand-crafted temporal statistics:
  For each timestep × each band: mean, std, min, max across field pixels
  Plus global temporal stats: mean/std/max/min/delta across all dates
  → Final vector size: T*C*4 + C*5

This is the most informative feature vector achievable without learned
temporal representations — a strong baseline for the thesis comparison.

USAGE
-----
  cd /home/johan/Thesis
  python run_baselines.py

  # Skip cross-validation (faster):
  python run_baselines.py --no_cv

  # Compare against a saved PSENet result:
  python run_baselines.py --psenet_results output/results/psenet_test_metrics.json

OUTPUT
------
  {output_path}/results/baseline_results.csv    — per-model metrics
  {output_path}/results/comparison_table.csv    — PSENet vs baselines
================================================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = Path('/home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset')
SAMPLES_FILE = DATASET_PATH / 'samples.json'
OUTPUT_PATH  = Path('/home/johan/Thesis/output/results')


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(field_dir: Path, timestamps: list, modality: str = 's1s2') -> np.ndarray:
    """
    Build a flat feature vector for one field.

    Per timestep: mean, std, min, max of each band across field pixels → (T * C * 4)
    Temporal:     mean, std, max, min, delta of per-date means across time → (C * 5)

    Args:
        field_dir:  path to field directory in dataset
        timestamps: list of ISO date strings from metadata.json
        modality:   's1s2' | 's1' | 's2'

    Returns:
        flat feature vector (T*C*4 + C*5,)
    """
    mask_path = field_dir / f'mask_{field_dir.name}.tif'
    mask = tifffile.imread(mask_path).astype(bool)

    per_date = []
    for ts in timestamps:
        bands = []

        if modality in ('s1s2', 's1'):
            s1 = tifffile.imread(field_dir / f's1_{ts}.tif')
            if s1.ndim == 3 and s1.shape[-1] == 6:
                s1 = s1.transpose(2, 0, 1)
            pixels_s1 = s1[:, mask].T.astype(np.float32)
            # Normalise Entropy and Alpha (same as dataset loader)
            pixels_s1[:, 2] = np.clip(pixels_s1[:, 2] / 100.0, 0, 1)
            pixels_s1[:, 4] = np.clip(pixels_s1[:, 4] / (np.pi / 2), 0, 1)
            bands.append(pixels_s1)

        if modality in ('s1s2', 's2'):
            s2 = tifffile.imread(field_dir / f's2_{ts}.tif')
            if s2.ndim == 3 and s2.shape[-1] == 6:
                s2 = s2.transpose(2, 0, 1)
            pixels_s2 = np.clip(s2[:, mask].T.astype(np.float32) / 10000.0, 0, 1)
            bands.append(pixels_s2)

        pixels = np.concatenate(bands, axis=1)   # (N, C)
        date_features = np.concatenate([
            pixels.mean(axis=0),
            pixels.std(axis=0),
            pixels.min(axis=0),
            pixels.max(axis=0),
        ])   # (C*4,)
        per_date.append(date_features)

    per_date = np.stack(per_date, axis=0)   # (T, C*4)
    C4 = per_date.shape[1]
    C  = C4 // 4
    per_date_means = per_date[:, :C]   # (T, C) — just the means

    temporal = np.concatenate([
        per_date_means.mean(axis=0),                          # temporal mean
        per_date_means.std(axis=0),                           # temporal variability
        per_date_means.max(axis=0),                           # seasonal peak
        per_date_means.min(axis=0),                           # seasonal trough
        per_date_means[-1] - per_date_means[0],               # start→end delta
    ])   # (C*5,)

    return np.concatenate([per_date.ravel(), temporal])   # (T*C*4 + C*5,)


def build_feature_matrix(samples: list, modality: str = 's1s2') -> tuple:
    """
    Extract feature vectors for all fields in the sample list.

    Returns:
        X: (N_fields, n_features)
        y: (N_fields,)
        field_ids: list of str
    """
    X, y, ids = [], [], []
    n_failed = 0

    for s in samples:
        fid       = str(s['field_id'])
        field_dir = DATASET_PATH / 'data' / fid
        meta_path = field_dir / 'metadata.json'

        if not meta_path.exists():
            n_failed += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        timestamps = meta['timestamps']

        try:
            feat = extract_features(field_dir, timestamps, modality)
            X.append(feat)
            y.append(s['residue_label'])
            ids.append(fid)
        except Exception as e:
            print(f'  Warning: skipping {fid}: {e}')
            n_failed += 1

    if n_failed:
        print(f'  Skipped {n_failed} fields due to missing data')

    return np.array(X), np.array(y), ids


# ---------------------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------------------

def get_baselines() -> dict:
    return {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )),
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, random_state=42,
            )),
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(
                kernel='rbf', C=10.0, gamma='scale',
                class_weight='balanced', probability=True, random_state=42,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    baselines: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, pipeline in baselines.items():
        print(f'\n  Training {name}...')
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        oa      = accuracy_score(y_test, preds)
        f1      = f1_score(y_test, preds, average='binary', pos_label=1, zero_division=0)
        f1_mac  = f1_score(y_test, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = float('nan')

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        print(f'  {name}: OA={oa:.3f} F1={f1:.3f} F1_macro={f1_mac:.3f} AUC={auc:.3f}')
        print(classification_report(y_test, preds,
                                    target_names=['no_residue', 'residue'],
                                    zero_division=0))
        print(f'  Confusion matrix:')
        print(f'    pred→  no_res  res')
        print(f'    no_res  {cm[0,0]:5d}  {cm[0,1]:4d}')
        print(f'    res     {cm[1,0]:5d}  {cm[1,1]:4d}')

        rows.append({
            'Model':    name,
            'OA':       round(oa, 4),
            'F1':       round(f1, 4),
            'F1_macro': round(f1_mac, 4),
            'AUC':      round(auc, 4),
        })

    return pd.DataFrame(rows).set_index('Model')


def cross_validate_baselines(
    baselines: dict,
    X: np.ndarray, y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for name, pipeline in baselines.items():
        print(f'  Cross-validating {name} ({n_splits} folds)...')
        cv = cross_validate(
            pipeline, X, y, cv=skf,
            scoring=['accuracy', 'f1', 'roc_auc'],
            n_jobs=-1,
        )
        rows.append({
            'Model':       name,
            'OA (mean)':   round(cv['test_accuracy'].mean(), 4),
            'OA (std)':    round(cv['test_accuracy'].std(),  4),
            'F1 (mean)':   round(cv['test_f1'].mean(),       4),
            'F1 (std)':    round(cv['test_f1'].std(),         4),
            'AUC (mean)':  round(cv['test_roc_auc'].mean(),  4),
            'AUC (std)':   round(cv['test_roc_auc'].std(),   4),
        })
    return pd.DataFrame(rows).set_index('Model')


def build_comparison_table(
    baseline_df:     pd.DataFrame,
    psenet_metrics:  dict = None,
) -> pd.DataFrame:
    if psenet_metrics:
        pse_row = pd.DataFrame([{
            'Model':    'PSENet (ours)',
            'OA':       round(psenet_metrics['oa'],       4),
            'F1':       round(psenet_metrics['f1'],       4),
            'F1_macro': round(psenet_metrics['f1_macro'], 4),
            'AUC':      round(psenet_metrics['auc'],      4),
        }]).set_index('Model')
        return pd.concat([pse_row, baseline_df])
    return baseline_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--modality',        default='s1s2',
                   choices=['s1s2', 's1', 's2'])
    p.add_argument('--no_cv',           action='store_true',
                   help='Skip cross-validation (faster)')
    p.add_argument('--psenet_results',  type=str, default=None,
                   help='Path to PSENet test_metrics.json for comparison table')
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load samples
    with open(SAMPLES_FILE) as f:
        all_samples = json.load(f)

    train_samples = [s for s in all_samples if s['set'] == 0]
    test_samples  = [s for s in all_samples if s['set'] == 2]

    print(f'\n── Extracting features (modality={args.modality}) ──')
    print(f'  Train fields: {len(train_samples)}')
    X_train, y_train, _ = build_feature_matrix(train_samples, args.modality)
    print(f'  Train feature matrix: {X_train.shape}')

    print(f'\n  Test fields: {len(test_samples)}')
    X_test,  y_test,  _ = build_feature_matrix(test_samples, args.modality)
    print(f'  Test feature matrix: {X_test.shape}')

    baselines = get_baselines()

    print('\n── Test set evaluation ──')
    baseline_df = evaluate_on_test(baselines, X_train, y_train, X_test, y_test)
    print('\nBaseline results:')
    print(baseline_df.to_string())

    baseline_df.to_csv(OUTPUT_PATH / 'baseline_results.csv')
    print(f'\n✓ Saved → {OUTPUT_PATH}/baseline_results.csv')

    if not args.no_cv:
        print('\n── Cross-validation (full dataset) ──')
        all_samples_flat = [s for s in all_samples if s['set'] in (0, 1, 2)]
        X_all, y_all, _  = build_feature_matrix(all_samples_flat, args.modality)
        cv_df = cross_validate_baselines(baselines, X_all, y_all)
        print('\nCross-validation results:')
        print(cv_df.to_string())
        cv_df.to_csv(OUTPUT_PATH / 'baseline_cv_results.csv')
        print(f'✓ Saved → {OUTPUT_PATH}/baseline_cv_results.csv')

    # Comparison table
    psenet_metrics = None
    if args.psenet_results:
        with open(args.psenet_results) as f:
            psenet_data    = json.load(f)
            psenet_metrics = psenet_data.get('test_metrics')

    table = build_comparison_table(baseline_df, psenet_metrics)
    print('\n── Final comparison table ──')
    print(table.to_string())
    table.to_csv(OUTPUT_PATH / 'comparison_table.csv')
    print(f'\n✓ Saved → {OUTPUT_PATH}/comparison_table.csv')


if __name__ == '__main__':
    main()
