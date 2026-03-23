import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry import box
import geopandas as gpd


def calculate_intersection_area(large_bounds, small_bounds):
    """
    Calculate the intersection area between two raster bounds.

    Parameters:
    -----------
    large_bounds : tuple
        Bounds of large raster (minx, miny, maxx, maxy)
    small_bounds : tuple
        Bounds of small raster (minx, miny, maxx, maxy)

    Returns:
    --------
    float : Intersection area
    """
    large_geom = box(*large_bounds)
    small_geom = box(*small_bounds)

    intersection = large_geom.intersection(small_geom)
    return intersection.area


def get_intersection_mask(large_bounds, small_raster, small_transform):
    """
    Create a mask showing which pixels in the small raster are covered by the large raster.

    Parameters:
    -----------
    large_bounds : tuple
        Bounds of large raster (minx, miny, maxx, maxy)
    small_raster : rasterio.DatasetReader
        The small raster dataset
    small_transform : affine.Affine
        Transform of the small raster

    Returns:
    --------
    numpy.ndarray : Boolean mask (True where large raster covers small raster)
    """
    large_geom = box(*large_bounds)

    # Create mask: True where geometry exists, False elsewhere
    # We want True where large raster covers, so invert=False
    mask = geometry_mask(
        [large_geom],
        out_shape=(small_raster.height, small_raster.width),
        transform=small_transform,
        invert=True  # True inside geometry, False outside
    )

    return mask


def fill_from_multiple_rasters(embedding_files, mask_file, out_file, resampling_method):

    # Open the small file to get target parameters
    with rasterio.open(mask_file) as mask:
        target_transform = mask.transform
        target_height = mask.height
        target_width = mask.width
        target_crs = mask.crs
        target_bounds = mask.bounds
        target_profile = mask.profile.copy()

    nodata_value = np.nan

    # Calculate intersection areas for each large raster
    intersections = []
    for file in embedding_files:
        with rasterio.open(file) as embeddings:
            area = calculate_intersection_area(embeddings.bounds, target_bounds)
            assert area > 0
            intersections.append({'file': file, 'area': area, 'bounds': embeddings.bounds})

    # Sort by intersection area (largest first)
    intersections.sort(key=lambda x: x['area'], reverse=True)

    # Initialize output array with nodata
    # Get band count and dtype from first large raster
    with rasterio.open(intersections[0]['file']) as first_embedding:
        band_count = first_embedding.count
        dtype = first_embedding.dtypes[0]

    # Create output array filled with nodata
    output_data = np.full((band_count, target_height, target_width), nodata_value, dtype=dtype)

    # Create a mask to track which pixels have been filled
    filled_mask = np.zeros((target_height, target_width), dtype=bool)

    # Process each large raster in order of intersection area
    for idx, item in enumerate(intersections, 1):
        with rasterio.open(item['file']) as embeddings:

            # Read data from large raster
            embedding_data = embeddings.read()

            # Create temporary array for resampled data
            resampled_data = np.empty((embeddings.count, target_height, target_width), dtype=dtype)

            # Reproject/resample the data
            reproject(
                source=embedding_data,
                destination=resampled_data,
                src_transform=embeddings.transform,
                src_crs=embeddings.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method,
                src_nodata=embeddings.nodata,
                dst_nodata=nodata_value
            )

            # Get mask of where this large raster covers the small raster
            coverage_mask = get_intersection_mask(item['bounds'], rasterio.open(mask_file), target_transform)

            # Only fill pixels that haven't been filled yet AND are covered by this raster
            pixels_to_fill = coverage_mask & ~filled_mask

            # Fill in pixels
            for band_idx in range(band_count):
                output_data[band_idx][pixels_to_fill] = resampled_data[band_idx][pixels_to_fill]

            # Update filled mask
            filled_mask |= pixels_to_fill

    # Update profile for output
    target_profile.update({
        'driver': 'GTiff',
        'height': target_height,
        'width': target_width,
        'transform': target_transform,
        'crs': target_crs,
        'count': band_count,
        'dtype': dtype,
        'nodata': nodata_value
    })

    # Write the output
    with rasterio.open(out_file, 'w', **target_profile) as dst:
        dst.write(output_data)

    total_pixels = target_height * target_width
    assert np.sum(filled_mask) == total_pixels


def resample_to_match(embeddings_file, mask_file, out_file, resampling_method) -> None:

    # Open the small file to get target parameters
    with rasterio.open(mask_file) as mask:
        # Get the transform, shape, CRS, and other metadata
        target_transform = mask.transform
        target_height = mask.height
        target_width = mask.width
        target_crs = mask.crs
        target_profile = mask.profile.copy()

    # Open the large file
    with rasterio.open(embeddings_file) as embeddings:
        # Read the data from the large file
        embedding_data = embeddings.read()
        embeddings_crs = embeddings.crs
        embeddings_transform = embeddings.transform

        # Update profile for output
        target_profile.update({
            'driver': 'GTiff',
            'height': target_height,
            'width': target_width,
            'transform': target_transform,
            'crs': target_crs,
            'count': embeddings.count,  # Use band count from embeddings
            'dtype': embeddings.dtypes[0]
        })

        # Create output array
        resampled_data = np.empty((embeddings.count, target_height, target_width), dtype=embeddings.dtypes[0])

        # Reproject/resample the data
        reproject(
            source=embedding_data,
            destination=resampled_data,
            src_transform=embeddings_transform,
            src_crs=embeddings_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling_method
        )

    # Write the resampled data to output file
    with rasterio.open(out_file, 'w', **target_profile) as dst:
        dst.write(resampled_data)


def get_em_file(root_folder: Path, metadata) -> Path:
    lon, lat = metadata['tile_lon'], metadata['tile_lat']
    f = root_folder / 'global_0.1_degree_representation' / '2024' / f'grid_{lon}_{lat}' / f'grid_{lon}_{lat}_2024.tiff'
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--crs', type=str, required=False, default='EPSG:3006')
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)

    # Loading embeddings registry
    registry = gpd.read_parquet(dataset_path / 'embeddings_registry.parquet')

    # Loading all fields
    fields = gpd.read_parquet(fields_file).to_crs(args.crs)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'
        field_path = dataset_path / 'data' / field_id
        mask_file = field_path / f'mask_{field_id}.tif'
        out_file = field_path / f'embeddings_{field_id}.tif'

        # Loading bbox
        gdf_bbox = gpd.read_parquet(field_path / f'bbox_{field_id}.parquet').to_crs('EPSG:4326')
        bbox = gdf_bbox.geometry.iloc[0]

        # Check if any embedding tiles fully cover the bbox
        bbox_within_embeddings = registry.geometry.apply(lambda x: bbox.within(x))
        if bbox_within_embeddings.any():
            em_files = [get_em_file(dataset_path, registry[bbox_within_embeddings].iloc[0])]
        else:
            # Check for intersection and stitch intersecting tiles together
            bbox_inter_em = registry.geometry.apply(lambda x: bbox.intersects(x))
            inter_em_tiles = registry[bbox_inter_em]
            em_files = [get_em_file(dataset_path, inter_em_tiles.iloc[i]) for i in range(len(inter_em_tiles))]
        fill_from_multiple_rasters(em_files, mask_file, out_file, resampling_method=Resampling.nearest)
