from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm
import shutil

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer

import numpy as np

from geotessera import GeoTessera


def get_embedding_tiles(gdf: gpd.GeoDataFrame) -> List:
    all_tiles = []
    for i, row in tqdm(enumerate(gdf.itertuples()), total=len(gdf)):
        geom = row.geometry
        bbox = (geom.bounds)
        tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=2024)
        all_tiles.extend(tiles)
    all_tiles = list(set(all_tiles))
    return all_tiles


def extract_footprint(arr: np.ndarray, geotransform) -> Polygon:
    m, n = arr.shape[0], arr.shape[1]
    # Convert to coordinates
    x_res, y_res, x_origin, y_origin = geotransform.a, geotransform.e, geotransform.c, geotransform.f
    x_max, y_min = x_origin + n * x_res, y_origin + m * y_res
    corners = [(x_origin, y_origin), (x_max, y_origin), (x_max, y_min), (x_origin, y_min)]
    footprint = Polygon(corners)
    return footprint


# https://geotessera.readthedocs.io/en/latest/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--batch_size', type=int, default=10, help='Number of tiles to process at once')
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)

    # Loading all fields
    assert fields_file.exists() and fields_file.suffix == '.parquet'
    fields = gpd.read_parquet(fields_file)
    fields = fields.to_crs(epsg=4326)

    # Initialize the client
    cache_path = dataset_path / 'geotessera_cache'
    cache_path.mkdir(exist_ok=True)
    gt = GeoTessera(verify_hashes=False, cache_dir=str(cache_path))

    tiles_to_fetch = get_embedding_tiles(fields)

    print('Creating registry')
    registry = {'tile_lon': [], 'tile_lat': [], 'geometry': []}
    for i in tqdm(range(0, len(tiles_to_fetch), args.batch_size)):
        tiles_batch = tiles_to_fetch[i:i + args.batch_size]
        embeddings_batch = gt.fetch_embeddings(tiles_batch)

        for year, tile_lon, tile_lat, embedding_array, crs, geotransform in embeddings_batch:
            registry['tile_lon'].append(tile_lon)
            registry['tile_lat'].append(tile_lat)
            geom = extract_footprint(embedding_array, geotransform)
            transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            geom_projected = transform(transformer.transform, geom)
            registry['geometry'].append(geom_projected)

    registry = gpd.GeoDataFrame(registry, crs='EPSG:4326', geometry='geometry')
    registry_file = dataset_path / 'embeddings_registry.parquet'
    registry.to_parquet(registry_file)
    print(f'Embedding registry at {str(registry_file)}')

    print('Downloading embeddings')
    for tile in tiles_to_fetch:
        try:
            _ = gt.export_embedding_geotiffs(
                tiles_to_fetch=[tile],  # Process one at a time
                output_dir=str(dataset_path),
                bands=None,
                compress="lzw"
            )
        except Exception as e:
            print(f"Failed to process tile {tile}: {e}")
            continue

    shutil.rmtree(cache_path)