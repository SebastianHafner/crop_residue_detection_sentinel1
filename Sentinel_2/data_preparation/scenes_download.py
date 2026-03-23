from pathlib import Path
import argparse
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import phidown_helpers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--date_field', type=str, required=False)
    parser.add_argument('--time_offset_weeks', type=int, required=False, default=4)
    parser.add_argument('--cloud_threshold', type=float, required=False, default=0.2)
    args = parser.parse_args()
    fields_file, out_path = Path(args.fields_file), Path(args.out_path)

    # Loading all fields
    assert fields_file.exists() and fields_file.suffix == '.parquet'
    fields = gpd.read_parquet(fields_file)
    fields = fields.to_crs(epsg=4326)

    # Load scene footprints
    footprints_file = out_path / 'scenes_metadata.parquet'
    if not footprints_file.exists():
        footprints = gpd.GeoDataFrame({'scene_id': [], 'geometry': []}, crs=4326)
    else:
        footprints = gpd.read_parquet(footprints_file)

    # Iterating over fields
    for field in tqdm(fields.itertuples(), total=len(fields)):

        # Extracting bounding box of field
        field_geom = field.geometry
        field_bbox = box(*field_geom.bounds)

        # Check if field is contained in any scene footprints
        covering_scenes = footprints.geometry.apply(lambda x: field_bbox.within(x))

        # If the field is not covered by any scenes, download a new scene covering the field
        if not covering_scenes.any():
            scene_id = phidown_helpers.get_scene(field_bbox, '2024-06-01', time_offset_weeks=2)

            if not scene_id in footprints['scene_id']:
                phidown_helpers.download_scene(scene_id, out_path)

                # Add to scene metadata
                footprint, epsg = phidown_helpers.get_scene_data_footprint(out_path / scene_id)
                metadata_entry = gpd.GeoDataFrame({'scene_id': [scene_id], 'geometry': [footprint]}, crs=epsg).to_crs(epsg=4326)
                footprints = gpd.GeoDataFrame(pd.concat([footprints, metadata_entry], ignore_index=True))
                footprints.to_parquet(footprints_file)

                # Convert to .tif file
                phidown_helpers.raw2tiff(out_path, scene_id, out_path)
