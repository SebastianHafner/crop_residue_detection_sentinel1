#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
s1_field_extraction.py
================================================================================

WHAT IT DOES
------------
For each field and each dated S1 mosaic, reprojects all S1 products to the
field mask grid and writes a single stacked GeoTIFF per date with a fixed
band order that matches exactly what crop_dataset.py and the PSENet model
expect.

Output file per field per date:
  s1_{YYYYMMDD}_{field_id}.tif — 6 bands (float32):
    Band 1: VV          — gamma0 backscatter (dB), from bs mosaic
    Band 2: VH          — gamma0 backscatter (dB), from bs mosaic
    Band 3: Alpha       — normalised 0-1 on disk (band 1 of pol mosaic)
    Band 4: Anisotropy  — normalised 0-1 on disk (band 2 of pol mosaic)
    Band 5: Entropy     — normalised 0-1 on disk (band 3 of pol mosaic)
    Band 6: DpRVI       — normalised 0-1 on disk (band 1 of dprvi mosaic)

When concatenated with S2 [B2,B3,B4,B8,B11,B12] in crop_dataset.py the full
12-band input vector to PSENet is:
  [VV, VH, Alpha, Anisotropy, Entropy, DpRVI, B2, B3, B4, B8, B11, B12]

Also writes a single metadata.json to the dataset root:
  {"timestamps": ["2024-06-07", "2024-06-19", ...]}
  Key is lowercase "timestamps" — matches crop_dataset.py meta['timestamps'].

Only dates where ALL three products (bs, pol, dprvi) are available are
included in the output and in metadata.json.

USAGE
-----
  python s1_field_extraction.py \
    --fields_file  /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet \
    --dataset_path /home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset \
    --s1_root      /home/johan/OST_processing/out_timeseries \
    --id_field     field_id

  # Overwrite already-existing output TIFs (e.g. after fixing band order):
  python s1_field_extraction.py ... --overwrite

REQUIREMENTS
------------
  pip install rasterio geopandas numpy shapely tqdm

OUTPUT
------
  dataset/metadata.json
  dataset/data/{field_id}/s1_{date}_{field_id}.tif   (6-band float32)
================================================================================
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fixed band order
# ---------------------------------------------------------------------------
# Each entry is (product_key, 1-indexed band in that product's mosaic TIF).
# pol band order verified with rasterio.descriptions:
#   band 1 = Alpha, band 2 = Anisotropy, band 3 = Entropy

PRODUCT_BAND_MAP = [
    ("bs",    1),   # Band 1 → VV
    ("bs",    2),   # Band 2 → VH
    ("pol",   1),   # Band 3 → Alpha
    ("pol",   2),   # Band 4 → Anisotropy
    ("pol",   3),   # Band 5 → Entropy
    ("dprvi", 1),   # Band 6 → DpRVI
]

BAND_NAMES = ["VV", "VH", "Alpha", "Anisotropy", "Entropy", "DpRVI"]
N_BANDS    = len(PRODUCT_BAND_MAP)   # 6


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def intersection_area(bounds_a, bounds_b) -> float:
    return box(*bounds_a).intersection(box(*bounds_b)).area


def coverage_mask(source_bounds, target_height, target_width,
                  target_transform) -> np.ndarray:
    """Boolean array: True where source_bounds covers the target grid."""
    return geometry_mask(
        [box(*source_bounds)],
        out_shape=(target_height, target_width),
        transform=target_transform,
        invert=True,
    )


# ---------------------------------------------------------------------------
# Reproject one source band onto the target grid
# ---------------------------------------------------------------------------

def reproject_band(src_path: Path, band_idx: int,
                   target_transform, target_crs,
                   target_height: int, target_width: int) -> np.ndarray:
    """
    Reproject a single band from src_path onto the target grid.

    Args:
        src_path:         source GeoTIFF
        band_idx:         1-indexed band number to read
        target_transform: affine transform of the target grid
        target_crs:       CRS of the target grid
        target_height:    pixel height of target grid
        target_width:     pixel width of target grid

    Returns:
        (target_height, target_width) float32 array, NaN where no data
    """
    with rasterio.open(src_path) as src:
        src_data  = src.read(band_idx)
        src_nodata = src.nodata
        resampled = np.full(
            (target_height, target_width), np.nan, dtype=np.float32
        )
        reproject(
            source        = src_data,
            destination   = resampled,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = target_transform,
            dst_crs       = target_crs,
            resampling    = Resampling.bilinear,
            src_nodata    = src_nodata,
            dst_nodata    = np.nan,
        )
    return resampled


# ---------------------------------------------------------------------------
# Stack all S1 bands into one GeoTIFF
# ---------------------------------------------------------------------------

def stack_s1_bands(product_files: dict, mask_file: Path, out_file: Path):
    """
    Reproject and stack all 6 S1 bands into a single GeoTIFF aligned to
    the field mask grid.

    Args:
        product_files: dict  product_key -> Path to mosaic TIF
                       e.g. {'bs': Path(...), 'pol': Path(...), 'dprvi': Path(...)}
        mask_file:     Path to field mask GeoTIFF (defines the target grid)
        out_file:      Path for output 6-band stacked GeoTIFF
    """
    with rasterio.open(mask_file) as mask_src:
        target_transform = mask_src.transform
        target_height    = mask_src.height
        target_width     = mask_src.width
        target_crs       = mask_src.crs
        target_profile   = mask_src.profile.copy()

    output = np.full(
        (N_BANDS, target_height, target_width), np.nan, dtype=np.float32
    )

    for out_band_idx, (product_key, src_band_idx) in enumerate(PRODUCT_BAND_MAP):
        if product_key not in product_files:
            raise ValueError(
                f"Missing product '{product_key}' needed for output "
                f"band {out_band_idx + 1} ({BAND_NAMES[out_band_idx]})"
            )

        band_data = reproject_band(
            src_path         = product_files[product_key],
            band_idx         = src_band_idx,
            target_transform = target_transform,
            target_crs       = target_crs,
            target_height    = target_height,
            target_width     = target_width,
        )
        output[out_band_idx] = band_data

    # Write stacked GeoTIFF
    target_profile.update({
        "driver":    "GTiff",
        "height":    target_height,
        "width":     target_width,
        "transform": target_transform,
        "crs":       target_crs,
        "count":     N_BANDS,
        "dtype":     "float32",
        "nodata":    np.nan,
        "compress":  "deflate",
    })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_file, "w", **target_profile) as dst:
        dst.write(output)
        # Self-documenting band descriptions
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.update_tags(i, name=name)


# ---------------------------------------------------------------------------
# Discover dated mosaics
# ---------------------------------------------------------------------------

def discover_dated_mosaics(merged_dir: Path, product_key: str) -> dict:
    """
    Find all per-date mosaics for a product key.

    Dated mosaics follow the naming convention:
      YYYYMMDD_{product_key}_SWEREF99TM_10m.tif

    Returns:
        dict: date_str (YYYYMMDD) -> Path, sorted by date
    """
    pattern   = f"*_{product_key}_SWEREF99TM_10m.tif"
    all_files = sorted(merged_dir.glob(pattern))

    dated = {}
    for f in all_files:
        prefix = f.stem.split("_")[0]
        if prefix.isdigit() and len(prefix) == 8:
            dated[prefix] = f

    return dict(sorted(dated.items()))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata_json(out_path: Path, date_strings: list):
    """
    Write metadata.json with sorted ISO timestamps.

    Uses lowercase key 'timestamps' to match crop_dataset.py:
      meta['timestamps']
    """
    timestamps = sorted(
        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
        for d in date_strings
    )
    with open(out_path, "w") as f:
        json.dump({"timestamps": timestamps}, f, indent=2)
    print(f"  Written: {out_path}  ({len(timestamps)} timestamps)")
    for ts in timestamps:
        print(f"    {ts}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="S1 time series -> per-field 6-band stacked GeoTIFFs"
    )
    p.add_argument("--fields_file",  required=True,
                   help="GeoParquet with field polygons")
    p.add_argument("--dataset_path", required=True,
                   help="Root dataset folder")
    p.add_argument("--s1_root",      required=True,
                   help="out_root used when running S1 processing scripts")
    p.add_argument("--id_field",     default=None,
                   help="Column name to use as field ID")
    p.add_argument("--crs",          default="EPSG:3006",
                   help="Target CRS (default: EPSG:3006 SWEREF99TM)")
    p.add_argument("--overwrite",    action="store_true",
                   help="Reprocess and overwrite existing output TIFs")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    s1_root      = Path(args.s1_root)
    merged_dir   = s1_root / "merged_SWEREF99TM_10m"

    if not merged_dir.exists():
        raise FileNotFoundError(
            f"merged_SWEREF99TM_10m not found at {merged_dir}\n"
            f"Run run_haalpha_timeseries.py (and backscatter/DpRVI scripts) first."
        )

    # ── Discover dated mosaics ─────────────────────────────────────────────
    print("\n── Discovering dated mosaics ──")
    product_keys    = ["bs", "pol", "dprvi"]
    product_mosaics = {}

    for key in product_keys:
        mosaics = discover_dated_mosaics(merged_dir, key)
        product_mosaics[key] = mosaics
        if mosaics:
            print(f"  {key:6s}: {len(mosaics)} dates  "
                  f"{list(mosaics.keys())}")
        else:
            print(f"  {key:6s}: [WARN] no dated mosaics found in {merged_dir}")

    # Only process dates where ALL three products are present
    sets_of_dates = [set(v.keys()) for v in product_mosaics.values() if v]
    if not sets_of_dates:
        raise RuntimeError("No dated mosaics found for any product.")

    complete_dates = sorted(set.intersection(*sets_of_dates))
    all_dates      = set.union(*sets_of_dates)
    missing_dates  = all_dates - set(complete_dates)

    if missing_dates:
        print(f"\n  [WARN] {len(missing_dates)} date(s) missing at least one "
              f"product — skipped: {sorted(missing_dates)}")

    print(f"\n  Complete dates (all 3 products present): {len(complete_dates)}")
    for d in complete_dates:
        print(f"    {d}")

    if not complete_dates:
        raise RuntimeError(
            "No dates have all three products (bs, pol, dprvi). "
            "Check that all processing scripts have been run."
        )

    # ── Write metadata.json ────────────────────────────────────────────────
    print("\n── Writing metadata.json ──")
    write_metadata_json(
        out_path     = dataset_path / "metadata.json",
        date_strings = complete_dates,
    )

    # ── Load fields ────────────────────────────────────────────────────────
    print("\n── Loading fields ──")
    fields = gpd.read_parquet(args.fields_file).to_crs(args.crs)
    print(f"  {len(fields)} fields in {args.crs}")

    # ── Per-field extraction ───────────────────────────────────────────────
    print("\n── Extracting per-field S1 stacks ──")
    print(f"  Band order: {' | '.join(BAND_NAMES)}")
    print(f"  Output:     s1_{{date}}_{{field_id}}.tif  (6 bands, float32)\n")

    n_ok = n_skip = n_err = 0

    for i, field in tqdm(enumerate(fields.itertuples()),
                         total=len(fields), desc="Fields"):
        field_id   = (
            str(getattr(field, args.id_field))
            if args.id_field else f"field{i:05d}"
        )
        field_path = dataset_path / "data" / field_id
        mask_file  = field_path / f"mask_{field_id}.tif"

        if not mask_file.exists():
            tqdm.write(f"[WARN] No mask for {field_id} — skipping")
            n_err += 1
            continue

        for date_str in complete_dates:
            out_file = field_path / f"s1_{date_str}_{field_id}.tif"

            if out_file.exists() and not args.overwrite:
                n_skip += 1
                continue

            # Build product_key -> mosaic Path dict for this date
            product_files = {
                key: product_mosaics[key][date_str]
                for key in product_keys
            }

            try:
                stack_s1_bands(
                    product_files = product_files,
                    mask_file     = mask_file,
                    out_file      = out_file,
                )
                n_ok += 1
            except Exception as e:
                tqdm.write(f"[ERROR] {field_id} / {date_str}: {e}")
                n_err += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n── Done ──")
    print(f"  Written:  {n_ok}")
    print(f"  Skipped:  {n_skip}  (already exist, use --overwrite to redo)")
    print(f"  Errors:   {n_err}")
    print(f"\n  Output: {dataset_path / 'data'}")


if __name__ == "__main__":
    main()