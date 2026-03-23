from typing import List
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import geopandas as gpd

import openeo
import xarray as xr
import io
import numpy as np

# Harvest date detection: https://doi.org/10.1016/j.rse.2025.115016

SPECTRAL_BANDS = ['b02', 'b03', 'b04', 'b08']


def get_valid_observation_timestamps(datacube: xr.DataArray, threshold_cloud_percentage: int = 20) -> List[str]:
    """
    Analyze SCL datacube and return timestamps of valid observations.
    """
    cloud_ids = [8, 9, 10]
    no_data_ids = [0, -9999]
    _, t, h, w = datacube.shape
    n_pixels = h * w
    scl_data = datacube.sel(variable='scl')

    # Ensure full coverage
    nodata_counts = scl_data.isin(no_data_ids).sum(dim=['y', 'x']).values
    full_coverage = np.array(nodata_counts) == 0

    # Ensure cloudy pixel percentage lower than threshold
    cloud_counts = scl_data.isin(cloud_ids).sum(dim=['y', 'x']).values
    cloud_free_obs = (np.array(cloud_counts) / n_pixels * 100) <= threshold_cloud_percentage

    valid_observations = np.logical_and(full_coverage, cloud_free_obs)
    valid_indices = [i for i, valid in enumerate(list(valid_observations)) if valid]

    # Extract timestamps for valid observations
    valid_timestamps = [np.datetime_as_string(datacube['t'].values[i], unit='D') for i in valid_indices]

    return valid_timestamps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--months', nargs='+', required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--cloud_threshold', type=int, required=False, default=50)
    parser.add_argument('--crs', type=str, required=False, default='EPSG:3006')
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)
    months = [int(m) for m in args.months]

    # Connect to digital earth sweden
    connection = openeo.connect('https://openeo.digitalearth.se')
    connection.authenticate_basic(username='testuser', password='secretpassword')

    # Loading all fields
    fields = gpd.read_parquet(fields_file)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'
        field_path = dataset_path / 'data' / field_id
        mask_file = field_path / f'mask_{field_id}.tif'

        # Loading bbox
        gdf_bbox = gpd.read_parquet(field_path / f'bbox_{field_id}.parquet').to_crs('EPSG:4326')
        bbox = gdf_bbox.geometry.iloc[0]

        # Get full temporal extent
        start_date = f'2024-{min(months):02d}-01'
        end_date = f'2024-{max(months) + 1:02d}-01'

        all_valid_timestamps = []

        for m in months:
            month_start = f'2024-{m:02d}-01'
            month_end = f'2024-{m + 1:02d}-01'

            # === STEP 1: Load and download ONLY SCL data (lightweight) ===
            datacube_scl = connection.load_collection(
                's2_msi_l2a',
                spatial_extent=bbox,
                temporal_extent=(month_start, month_end),
                bands=['scl'],
                properties={'eo:cloud_cover': lambda val: val < args.cloud_threshold},
            )

            netcdf_data_scl = datacube_scl.download(format='NetCDF')
            dataset_scl = xr.open_dataset(io.BytesIO(netcdf_data_scl))
            obs_scl_array = dataset_scl.to_array()

            # Identify valid observation timestamps
            valid_timestamps = get_valid_observation_timestamps(obs_scl_array)

            if not valid_timestamps:
                print(f"No valid observations found for field {field_id} in month {m}")
                continue

            # === STEP 2: Load datacube once, then filter to valid dates before download ===
            # Create the spectral datacube (lazy - no download yet)
            datacube_spectral = connection.load_collection(
                's2_msi_l2a',
                spatial_extent=bbox,
                temporal_extent=(month_start, month_end),
                bands=SPECTRAL_BANDS,
                properties={'eo:cloud_cover': lambda val: val < args.cloud_threshold},
            )

            # For each valid timestamp, filter and download as GeoTIFF
            for timestamp in valid_timestamps:
                # Create date range for this specific day
                date_start = timestamp
                date_end_dt = np.datetime64(timestamp, 'D') + np.timedelta64(1, 'D')
                date_end = np.datetime_as_string(date_end_dt, unit='D')

                # Filter datacube to this specific date (still lazy)
                obs_spectral_filtered = datacube_spectral.filter_temporal(date_start, date_end)

                # Download directly as GeoTIFF - no intermediate format needed!
                try:
                    output_file = field_path / f's2_{field_id}_{timestamp}.tif'
                    geotiff_data = obs_spectral_filtered.download(format='GTiff')

                    # Write the GeoTIFF data directly to file
                    with open(output_file, 'wb') as f:
                        f.write(geotiff_data)

                    all_valid_timestamps.append(timestamp)

                except Exception as e:
                    print(f"Error downloading data for {timestamp}: {e}")
                    continue

        field_metadata = {'timestamps': all_valid_timestamps}
        with open(field_path / f's2_metadata_{field_id}.json', 'w') as f:
            json.dump(field_metadata, f, indent=2)