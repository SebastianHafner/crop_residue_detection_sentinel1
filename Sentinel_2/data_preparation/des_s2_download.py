from typing import Tuple, List
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import geopandas as gpd
import rasterio
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box

from datetime import datetime, timedelta

import openeo
import xarray as xr
import io
import numpy as np

import helpers


def get_s2_datacube(geom: BaseGeometry, timeframe: Tuple[str, str], thresh_cloud_cover: int,
                    bands: List[str]) -> xr.DataArray:
    datacube = connection.load_collection(
        's2_msi_l2a',
        spatial_extent=geom,
        temporal_extent=timeframe,
        bands=bands,
        properties = {'eo:cloud_cover':  lambda val: val < thresh_cloud_cover},
    )

    netcdf_data = datacube.download(format='NetCDF')

    # Load the NetCDF data into a xarray.Dataset directly from memory
    dataset = xr.open_dataset(io.BytesIO(netcdf_data))

    # If you prefer working with a DataArray, you can convert the Dataset to a DataArray
    data_array = dataset.to_array()

    return data_array


# Returns index of best S2 observation among the candidate scenes
def get_best_field_observation(datacube: xr.DataArray) -> Tuple[int, Tuple[int, int]]:
    cloud_ids = [8, 9, 10]
    no_data_ids = [0, -9999]

    scl_data = datacube.sel(variable='scl')
    cloud_counts = scl_data.isin(cloud_ids).sum(dim=['y', 'x']).values
    nodata_counts = scl_data.isin(no_data_ids).sum(dim=['y', 'x']).values

    full_coverage = np.array(nodata_counts) == 0

    masked_numbers = np.where(full_coverage, cloud_counts, np.inf)
    index = np.argmin(masked_numbers)
    # print(cloud_counts, nodata_counts, index)

    return int(index), (nodata_counts[index], cloud_counts[index])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--date_field', type=str, required=False)
    parser.add_argument('--time_offset_weeks', type=int, required=False, default=4)
    parser.add_argument('--cloud_threshold', type=int, required=False, default=80)
    parser.add_argument('--crs', type=str, required=False, default='EPSG:3006')
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)

    # Connect to digital earth sweden
    connection = openeo.connect('https://openeo.digitalearth.se')
    connection.authenticate_basic(username='testuser', password='secretpassword')

    # Loading all fields
    assert fields_file.exists() and fields_file.suffix == '.parquet'
    fields = gpd.read_parquet(fields_file)
    fields = fields.to_crs(epsg=4326)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'

        # Extracting bounding box of field
        field_geom = field.geometry
        field_geom_reproj = helpers.reproject_geometry(field_geom, str(fields.crs), args.crs)

        field_bbox_reproj = box(*field_geom_reproj.bounds)
        field_bbox_reproj_buffered = field_bbox_reproj.buffer(10)
        field_bbox_buffered = helpers.reproject_geometry(field_bbox_reproj_buffered, args.crs, str(fields.crs))

        out_path = dataset_path / field_id
        out_path.mkdir(exist_ok=True)
        gdf_field = gpd.GeoDataFrame({'geometry': [field_geom_reproj]}, crs=args.crs)
        gdf_field.to_parquet(out_path / f'polygon_{field_id}.parquet')

        start_date = '2024-07-01'  # This will come from the fields file
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(weeks=args.time_offset_weeks)).strftime('%Y-%m-%d')

        cand_obs = get_s2_datacube(field_bbox_buffered, (start_date, end_date), args.cloud_threshold,
                                   bands=['scl'])
        index, best_scene_metadata = get_best_field_observation(cand_obs)
        timestamp = np.datetime_as_string(cand_obs['t'].values[index], unit='D')

        cand_obs = get_s2_datacube(field_bbox_buffered, (start_date, end_date), args.cloud_threshold,
                                   bands=['b02', 'b03', 'b04', 'b08'])

        obs = cand_obs.isel(t=0)

        # Set the spatial dimensions and CRS
        obs = obs.rio.write_crs(args.crs)

        # Save as GeoTIFF
        output_file = out_path / f's2_{field_id}.tif'
        obs.rio.to_raster(output_file)

        # Creating mask
        with rasterio.open(output_file) as src:
            mask_meta = src.meta.copy()
            mask_meta.update({
                "count": 1,
                "dtype": "uint8",
                "nodata": None,
            })

            mask_arr = helpers.get_fully_contained_mask(field_geom_reproj, src.transform, src.height, src.width)
            with rasterio.open(out_path / f'mask_{field_id}.tif', "w", **mask_meta) as dest:
                dest.write(mask_arr, 1)

        field_metadata = {
            's2_timestamp': timestamp,
            'valid_pixel_percentage': (best_scene_metadata[0] / obs.size) * 100,
            'cloudy_pixel_percentage': (best_scene_metadata[1] / obs.size) * 100,
        }

        with open(out_path / f'metadata_{field_id}.json', 'w') as f:
            json.dump(field_metadata, f, indent=2)

