#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for the split Sentinel-1 SLC processing scripts.

Used by: run_backscatter.py, run_haalpha.py, run_dprvi.py

Key fix: SNAP outputs 0 for background/nodata pixels. All conversion and
reprojection functions now convert 0 → NaN so that the merge correctly
treats background as transparent, eliminating gaps between subswaths and
black frames around bursts.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
import fnmatch
import xml.etree.ElementTree as eTree

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed.")

from ost.s1.s1scene import Sentinel1Scene as S1Scene
from ost.s1.burst_inventory import burst_extract


TARGET_EPSG = 3006
TARGET_RES = 10


def set_target(epsg, res):
    global TARGET_EPSG, TARGET_RES
    TARGET_EPSG = epsg
    TARGET_RES = res


def wsl_unc_to_linux(path_str):
    p = path_str.strip()
    if p.lower().startswith("\\\\wsl.localhost\\"):
        parts = p.split("\\")
        if len(parts) >= 5:
            return "/" + "/".join(parts[4:])
    return p


def find_one_zip(scene_dir):
    zips = sorted(scene_dir.glob("S1*_SLC_*.zip"))
    if not zips:
        zips = sorted(scene_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files in: {scene_dir}")
    if len(zips) > 1:
        print(f"[INFO] Found {len(zips)} zips. Using: {zips[0].name}")
    return zips[0]


def extract_burst_inventory_from_zip(scene_zip, scene_id, rel_orbit, start_date):
    import geopandas as gpd
    if not scene_zip.exists():
        raise FileNotFoundError(f"Zip not found: {scene_zip}")
    columns = ["SceneID", "Track", "Date", "SwathID", "AnxTime", "BurstNr", "geometry"]
    gdf_final = gpd.GeoDataFrame(columns=columns, crs="epsg:4326")
    archive = zipfile.ZipFile(scene_zip, "r")
    for anno_file in fnmatch.filter(archive.namelist(), "*/annotation/s*.xml"):
        gdf = burst_extract(scene_id, rel_orbit, start_date,
                            eTree.parse(archive.open(anno_file)))
        gdf_final = pd.concat([gdf_final, gdf])
    return gdf_final.drop_duplicates(["AnxTime"], keep="first")


def resolve_scene_zip(args):
    if args.scene_zip:
        return Path(wsl_unc_to_linux(args.scene_zip)).expanduser().resolve()
    elif args.scene_dir:
        return find_one_zip(
            Path(wsl_unc_to_linux(args.scene_dir)).expanduser().resolve())
    else:
        print("ERROR: Provide --scene_zip or --scene_dir")
        sys.exit(2)


# ---------------------------------------------------------------------------
# AOI filtering — keep only bursts that overlap the study area
# ---------------------------------------------------------------------------

def filter_bursts_by_aoi(burst_gdf, aoi_file):
    """
    Filter burst inventory to only include bursts that overlap the AOI.

    Parameters
    ----------
    burst_gdf : GeoDataFrame
        Full burst inventory (EPSG:4326)
    aoi_file : str or Path
        Path to a parquet/shapefile with field polygons.
        The convex hull of all geometries is used as the AOI.

    Returns
    -------
    GeoDataFrame : filtered bursts that intersect the AOI
    """
    import geopandas as gpd

    aoi_path = Path(aoi_file)
    if aoi_path.suffix == ".parquet":
        fields = gpd.read_parquet(aoi_path)
    else:
        fields = gpd.read_file(aoi_path)

    # Reproject to WGS84 to match burst inventory
    if fields.crs and fields.crs.to_epsg() != 4326:
        fields = fields.to_crs(4326)

    aoi_geom = fields.geometry.unary_union.convex_hull

    # Filter: keep bursts whose geometry intersects the AOI
    mask = burst_gdf.geometry.intersects(aoi_geom)
    filtered = burst_gdf[mask].copy()

    n_orig = len(burst_gdf)
    n_filt = len(filtered)
    print(f"[AOI] {n_filt}/{n_orig} bursts overlap study area "
          f"(skipping {n_orig - n_filt})")

    return filtered


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(scene_zip, out_root, temp_dir,
                 backscatter=False, haalpha=False, dprvi=False,
                 to_db=False, remove_speckle=False, create_ls_mask=False,
                 dprvi_window=3, product_type="RTC-gamma0", resolution=20):
    """Build a config dict with only the requested products enabled."""
    return {
        "download_dir": str(scene_zip.parent.parent.parent.parent.parent),
        "data_mount": str(scene_zip.parent.parent.parent.parent.parent),
        "processing_dir": str(out_root),
        "temp_dir": str(temp_dir),
        "snap_cpu_parallelism": 2,
        "subset": False,
        "aoi": None,
        "processing": {
            "single_ARD": {
                "image_type": "SLC",
                "ard_type": "OST-RTC",
                "resolution": resolution,
                "remove_border_noise": True,
                "product_type": product_type,
                "polarisation": "VV,VH",
                "coherence_bands": "VV, VH",
                "to_db": bool(to_db),
                "to_tif": False,
                "geocoding": "terrain",
                "backscatter": bool(backscatter),
                "H-A-Alpha": bool(haalpha),
                "DpRVI": bool(dprvi),
                "dprvi_window_size": dprvi_window,
                "coherence": False,
                "polarimetric": True,
                "remove_speckle": bool(remove_speckle),
                "remove_pol_speckle": bool(remove_speckle),
                "speckle_filter": {
                    "filter": "Refined Lee", "ENL": 1, "estimate_ENL": True,
                    "sigma": 0.9, "filter_x_size": 5, "filter_y_size": 5,
                    "window_size": "7x7", "target_window_size": "3x3",
                    "num_of_looks": 1, "damping": 2, "pan_size": 50
                },
                "pol_speckle_filter": {
                    "polarimetric_filter": "Refined Lee Filter",
                    "filter_size": 5, "num_of_looks": 1,
                    "window_size": "7x7", "target_window_size": "3x3",
                    "pan_size": 50, "sigma": 0.9
                },
                "create_ls_mask": bool(create_ls_mask),
                "apply_ls_mask": False,
                "dem": {
                    "dem_name": "Copernicus 30m Global DEM",
                    "dem_file": "", "dem_nodata": 0,
                    "dem_resampling": "BILINEAR_INTERPOLATION",
                    "image_resampling": "BILINEAR_INTERPOLATION",
                    "egm_correction": True,
                    "out_projection": 32633
                },
                "coherence_azimuth": 10,
                "coherence_range": 2
            },
            "time-series_ARD": {
                "to_db": True, "remove_mt_speckle": False,
                "apply_ls_mask": False,
                "mt_speckle_filter": {
                    "filter": "Refined Lee", "ENL": 1, "estimate_ENL": True,
                    "sigma": 0.9, "filter_x_size": 5, "filter_y_size": 5,
                    "window_size": "7x7", "target_window_size": "3x3",
                    "num_of_looks": 1, "damping": 2, "pan_size": 50
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# .dim → GeoTIFF  (converts SNAP's 0-nodata to NaN)
# ---------------------------------------------------------------------------

def dim_to_tif(dim_path, out_tif):
    """
    Convert SNAP .dim/.data to multi-band GeoTIFF.
    SNAP uses 0 for background — we convert 0 → NaN so merges work correctly.
    """
    data_dir = dim_path.with_suffix(".data")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No .data folder: {data_dir}")
    img_files = sorted(data_dir.glob("*.img"))
    if not img_files:
        raise FileNotFoundError(f"No .img bands in {data_dir}")

    with rasterio.open(img_files[0]) as src0:
        meta = src0.meta.copy()
    meta.update(driver="GTiff", count=len(img_files), dtype="float32",
                nodata=np.nan, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256)

    with rasterio.open(out_tif, "w", **meta) as dst:
        for i, img_f in enumerate(img_files, 1):
            with rasterio.open(img_f) as src:
                band = src.read(1).astype(np.float32)
                # Convert SNAP's 0-background to NaN
                band[band == 0] = np.nan
                dst.write(band, i)
            dst.set_band_description(i, img_f.stem)
    return out_tif


# ---------------------------------------------------------------------------
# Reproject  (preserves NaN nodata)
# ---------------------------------------------------------------------------

def reproject_to_sweref(in_tif, out_tif):
    """Reproject to target CRS. Ensures nodata=NaN is preserved."""
    dst_crs = f"EPSG:{TARGET_EPSG}"
    dst_res = TARGET_RES
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=(dst_res, dst_res))
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width,
                       height=height, dtype="float32", nodata=np.nan,
                       compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256)

        # Determine source nodata: use file's nodata, fall back to NaN
        src_nodata = src.nodata if src.nodata is not None else np.nan

        with rasterio.open(out_tif, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs,
                    src_nodata=src_nodata, dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                desc = src.descriptions[i - 1] if src.descriptions else None
                if desc:
                    dst.set_band_description(i, desc)
    return out_tif


# ---------------------------------------------------------------------------
# Merge bursts  (NaN-aware, no gaps)
# ---------------------------------------------------------------------------

def merge_burst_tifs(tif_list, out_tif):
    """
    Merge burst GeoTIFFs into scene mosaic.
    All inputs should already have NaN as nodata (from dim_to_tif / reproject).
    """
    valid = [t for t in tif_list if Path(t).exists()]
    if not valid:
        return None

    datasets = [rasterio.open(t) for t in valid]
    mosaic, out_transform = merge(datasets, nodata=np.nan, method="first")

    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
        transform=out_transform, count=mosaic.shape[0],
        dtype="float32", nodata=np.nan,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    )

    descriptions = []
    for i in range(1, datasets[0].count + 1):
        desc = datasets[0].descriptions[i - 1] if datasets[0].descriptions else None
        descriptions.append(desc)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mosaic)
        for i, desc in enumerate(descriptions, 1):
            if desc:
                dst.set_band_description(i, desc)

    for ds in datasets:
        ds.close()

    return out_tif


# ---------------------------------------------------------------------------
# Retry helper — resets failed bursts so they get reprocessed on next run
# ---------------------------------------------------------------------------

def reset_failed_bursts(out_root, product_key, final_glob_pattern):
    """
    Find bursts that don't have a final tif and reset their .processed marker.
    Returns the number of bursts reset.

    product_key: 'bs', 'pol', or 'dprvi'
    final_glob_pattern: e.g. '*pol_SWEREF99TM_10m.tif'
    """
    marker = {
        "bs": ".bs.processed",
        "pol": ".pol.processed",
        "dprvi": ".dprvi.processed",
    }[product_key]

    reset_count = 0
    for burst_dir in sorted(out_root.glob("S1*_IW*")):
        if not burst_dir.is_dir():
            continue
        has_final = any(burst_dir.rglob(f"final/{final_glob_pattern}"))
        if not has_final:
            for m in burst_dir.rglob(marker):
                m.unlink()
                reset_count += 1
    return reset_count