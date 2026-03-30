"""
================================================================================
scripts/build_samples_json.py
================================================================================

WHAT IT DOES
------------
Scans the per-field dataset directory, joins each field's ID with the SBA crop
residue survey labels (via the fields parquet), and writes a samples.json file
that the DataLoader consumes.  Also performs the train/val/test split.

Each entry in samples.json looks like:
  {
    "field_id": "12345",
    "residue_label": 1,          # 0 = no residue, 1 = residue
    "crop_type": "winter_wheat",
    "set": 0                     # 0=train, 1=val, 2=test
  }

USAGE
-----
  cd /home/johan/Thesis
  python scripts/build_samples_json.py

  # Custom split ratios:
  python scripts/build_samples_json.py --train 0.7 --val 0.15

  # Only keep winter wheat fields:
  python scripts/build_samples_json.py --crop_filter winter_wheat

REQUIREMENTS
------------
  pip install geopandas pandas numpy scikit-learn tqdm

OUTPUT
------
  /home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset/samples.json
================================================================================
"""

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — match config.yaml
# ---------------------------------------------------------------------------
DATASET_PATH   = Path('/home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset')
FIELDS_PARQUET = Path('/home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet')
OUTPUT_JSON    = DATASET_PATH / 'samples.json'

# Column names in the fields parquet — adjust if yours differ
FIELD_ID_COL      = 'field_id'
RESIDUE_LABEL_COL = 'residue_label'   # 0/1 integer column
CROP_TYPE_COL     = 'crop_type'       # string column, e.g. 'winter_wheat'

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Build samples.json for the DataLoader.')
    p.add_argument('--train',        type=float, default=0.70,  help='Train fraction')
    p.add_argument('--val',          type=float, default=0.15,  help='Val fraction')
    p.add_argument('--crop_filter',  type=str,   default='winter_wheat',
                   help='Only include this crop type. Pass "all" to skip filtering.')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def load_labels(fields_parquet: Path, crop_filter: str) -> pd.DataFrame:
    """Load field geometries + labels from the SBA-joined parquet."""
    gdf = gpd.read_parquet(fields_parquet)

    # Ensure field_id is string for consistent key matching
    gdf[FIELD_ID_COL] = gdf[FIELD_ID_COL].astype(str)

    # Filter to target crop type
    if crop_filter != 'all':
        before = len(gdf)
        gdf = gdf[gdf[CROP_TYPE_COL].str.lower() == crop_filter.lower()]
        print(f'  Crop filter "{crop_filter}": {before} → {len(gdf)} fields')

    # Drop rows missing labels
    missing = gdf[RESIDUE_LABEL_COL].isna().sum()
    if missing > 0:
        print(f'  Dropping {missing} fields with missing residue label')
        gdf = gdf.dropna(subset=[RESIDUE_LABEL_COL])

    gdf[RESIDUE_LABEL_COL] = gdf[RESIDUE_LABEL_COL].astype(int)
    return gdf[[FIELD_ID_COL, RESIDUE_LABEL_COL, CROP_TYPE_COL]]


def scan_dataset_fields(dataset_path: Path) -> list:
    """Return field IDs that exist in the dataset directory."""
    field_dirs = [d.name for d in (dataset_path / 'data').iterdir() if d.is_dir()]
    print(f'  Found {len(field_dirs)} field directories in dataset')
    return field_dirs


def build_samples(df: pd.DataFrame, dataset_fields: list,
                  train_frac: float, val_frac: float, seed: int) -> list:
    """
    Inner-join labels with dataset directories, then split into
    train (set=0) / val (set=1) / test (set=2) with stratification.
    """
    available = set(dataset_fields)
    df = df[df[FIELD_ID_COL].isin(available)].copy()
    print(f'  Fields with both label and data: {len(df)}')

    # Class distribution
    counts = df[RESIDUE_LABEL_COL].value_counts()
    print(f'  Class distribution:\n{counts.to_string()}')
    print(f'  Imbalance ratio: {counts.max()/counts.min():.1f}:1')

    # Stratified split
    test_frac = 1.0 - train_frac - val_frac
    ids       = df[FIELD_ID_COL].values
    labels    = df[RESIDUE_LABEL_COL].values

    ids_trainval, ids_test, y_trainval, _ = train_test_split(
        ids, labels, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_frac_of_trainval = val_frac / (train_frac + val_frac)
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=val_frac_of_trainval, stratify=y_trainval, random_state=seed
    )

    set_map = {fid: 0 for fid in ids_train}
    set_map.update({fid: 1 for fid in ids_val})
    set_map.update({fid: 2 for fid in ids_test})

    samples = []
    for _, row in df.iterrows():
        fid = row[FIELD_ID_COL]
        samples.append({
            'field_id':      fid,
            'residue_label': int(row[RESIDUE_LABEL_COL]),
            'crop_type':     str(row[CROP_TYPE_COL]),
            'set':           set_map[fid],
        })

    # Print split summary
    for split_id, split_name in [(0,'train'), (1,'val'), (2,'test')]:
        split = [s for s in samples if s['set'] == split_id]
        n_res = sum(s['residue_label'] for s in split)
        print(f'  {split_name:5s}: {len(split):4d} fields  '
              f'({n_res} residue / {len(split)-n_res} no-residue)')

    return samples


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('\n── Loading labels from parquet ──')
    df = load_labels(FIELDS_PARQUET, args.crop_filter)

    print('\n── Scanning dataset directory ──')
    dataset_fields = scan_dataset_fields(DATASET_PATH)

    print('\n── Building samples ──')
    samples = build_samples(df, dataset_fields, args.train, args.val, args.seed)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f'\n✓ Wrote {len(samples)} samples → {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
