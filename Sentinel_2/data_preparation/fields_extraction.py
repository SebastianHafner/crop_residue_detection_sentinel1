from pathlib import Path
from tqdm import tqdm

import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize

import geopandas as gpd
import argparse
from shapely.geometry import Polygon, box

from data_preparation import phidown_helpers, helpers
# cloud masking: https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/Sentinel-2.ipynb


def clip2field(raster_file: Path, field_geom: Polygon, field_epsg: str, field_id: str, out_path: Path) -> None:
    out_path = out_path / field_id
    out_path.mkdir(exist_ok=True)
    geom_file = out_path / f'{field_id}.parquet'
    if not geom_file.exists():
        gdf_field = gpd.GeoDataFrame({'geometry': [field_geom]}, crs=field_epsg)
        gdf_field.to_parquet(geom_file)

    with rasterio.open(raster_file) as src:
        raster_epsg = src.crs.to_epsg()

        # Reproject field if CRS mismatch
        if field_epsg != raster_epsg:
            field_geom = helpers.reproject_geometry(field_geom, field_epsg, raster_epsg)
            field_geom_buffer = field_geom.buffer(20)

        # --- Compute bounding box window in raster CRS ---
        minx, miny, maxx, maxy = field_geom_buffer.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)

        # --- Read clipped S2 data ---
        clipped = src.read(window=window)

        # --- Metadata for BOTH outputs ---
        out_transform = rasterio.windows.transform(window, src.transform)
        height = int(window.height)
        width = int(window.width)

        out_meta = src.meta.copy()
        out_meta.update({
            'height': height,
            'width': width,
            'transform': out_transform
        })

        ### 1. Save clipped S2 raster ###
        timestamp = phidown_helpers.get_timestamp(scene_id).strftime('%Y-%m-%d')
        clipped_path = out_path / f'{field_id}_{timestamp}.tif'
        with rasterio.open(clipped_path, "w", **out_meta) as dest:
            dest.write(clipped)

        mask_file = out_path / f'{field_id}_mask.tif'
        if not mask_file.exists():
            mask_arr = helpers.get_fully_contained_mask(field_geom, out_transform, height, width)
            mask_meta = out_meta.copy()
            mask_meta.update({
                "count": 1,
                "dtype": "uint8",
            })

            with rasterio.open(mask_file, "w", **mask_meta) as dest:
                dest.write(mask_arr, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', required=True)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
    fields_file, data_path = Path(args.fields_file), Path(args.data_path)
    fields_path = data_path / 'fields'
    fields_path.mkdir(exist_ok=True)

    # Loading scenes metadata
    metadata_file = data_path / 'scenes_metadata.parquet'
    assert metadata_file.exists() and metadata_file.suffix == '.parquet'
    metadata = gpd.read_parquet(metadata_file)

    # Loading all fields
    assert fields_file.exists() and fields_file.suffix == '.parquet'
    fields = gpd.read_parquet(fields_file)
    fields = fields.to_crs(epsg=4326)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_geom = field.geometry

        # Find intersecting scene
        field_bbox = box(*field_geom.bounds)
        is_covering_scene = metadata.geometry.apply(lambda x: field_bbox.within(x))
        assert is_covering_scene.any()
        scene_id = metadata[is_covering_scene].iloc[0]['scene_id']

        clip2field(data_path / f'{scene_id}.tif', field_geom, 'EPSG:4326', f'field{i}', fields_path)

        if i == 3:
            break