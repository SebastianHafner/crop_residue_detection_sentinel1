from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List
import cv2

from phidown.search import CopernicusDataSearcher

import rasterio
import geopandas as gpd
from shapely.geometry import box, Polygon


import numpy as np


BANDS = {
    '10m': ['B02', 'B03', 'B04', 'B08'],
}

# https://sentiwiki.copernicus.eu/web/s2-products

# Returns row
def get_scene(bbox: Polygon, harvest_date: str, time_offset_weeks: int) -> str:
    harvest_date = datetime.strptime(harvest_date, "%Y-%m-%d")
    start_date = harvest_date + timedelta(weeks=time_offset_weeks)
    # end_date = datetime(harvest_date.year, 12, 31, 23, 59, 59)
    end_date = start_date + timedelta(weeks=4)

    searcher = CopernicusDataSearcher()

    # Configure the search parameters
    searcher.query_by_filter(
        collection_name='SENTINEL-2',
        attributes={'processingLevel': 'S2MSI2A'},
        # cloud_cover_threshold=20,
        aoi_wkt=bbox.wkt,
        start_date=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
        end_date=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
        top=1000,
    )
    # attributes={'relativeOrbitNumber': '51'}

    df_s2 = searcher.execute_query()
    df_s2['cloud_cover'] = df_s2['Attributes'].apply(lambda x: float(x[2]['Value']))
    df_s2 = df_s2.sort_values('cloud_cover', ascending=True)

    return None if len(df_s2) == 0 else df_s2.iloc[0]['Name']


def scene_is_available(scene_id: str, out_path: Path) -> bool:
    out_folder = out_path / scene_id
    return out_folder.exists()


def get_scene_footprint(scene_folder: Path) -> Tuple[Polygon, int]:
    subfolder = [f for f in (scene_folder / 'GRANULE').iterdir() if f.is_dir()][0]
    band_file = [f for f in (subfolder / 'IMG_DATA' / 'R10m').iterdir() if f.is_file() and f.suffix == '.jp2'][0]
    # Read extent + CRS
    with rasterio.open(band_file) as src:
        left, bottom, right, top = src.bounds
        footprint = box(left, bottom, right, top)
        epsg_code = src.crs.to_epsg()
        # Transform polygon to WGS84 using pyproj
        # transformer = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
        # footprint = shp_transform(transformer.transform, footprint)
    return footprint, epsg_code


def get_scene_data_footprint(scene_folder: Path) -> Tuple[Polygon, int]:
    subfolder = [f for f in (scene_folder / 'GRANULE').iterdir() if f.is_dir()][0]
    band_file = [f for f in (subfolder / 'IMG_DATA' / 'R10m').iterdir() if f.is_file() and f.suffix == '.jp2'][0]

    with rasterio.open(band_file) as src:
        arr = src.read(1)

        # Find corners
        top, bottom = [None, None], [None, None]
        for j in range(0, arr.shape[1]):
            if arr[0, j] != 0 and top[0] is None:
                top[0] = j
            if arr[-1, j] != 0 and bottom[0] is None:
                bottom[0] = j
        for j in range(arr.shape[1] - 1, -1, -1):
            if arr[0, j] != 0 and top[1] is None:
                top[1] = j
            if arr[-1, j] != 0 and bottom[1] is None:
                bottom[1] = j
        corners = [[0, top[0]], [0, top[1]], [arr.shape[0] - 1, bottom[1]], [arr.shape[0] - 1, bottom[0]]]

        # Convert to coordinates
        transform = src.transform
        x_res, y_res, x_origin, y_origin = transform.a, transform.e, transform.c, transform.f
        corners = [(x_origin + j * x_res, y_origin + i * y_res) for i, j in corners]
        footprint = Polygon(corners)
        epsg_code = src.crs.to_epsg()

    return footprint, epsg_code


def download_scene(scene_id: str, out_folder: Path) -> None:
    searcher = CopernicusDataSearcher()
    searcher.download_product(
        scene_id,
        config_file='.s5cfg',
        output_dir=str(out_folder),
        show_progress=False,
    )


def get_relative_orbit_number(product_id: str) -> int:
    return int(product_id.split('_')[4][1:])


def get_timestamp(product_id: str) -> datetime:
    dt = datetime.strptime(product_id.split('_')[2], '%Y%m%dT%H%M%S')
    return dt


# TODO: Add cloud information to bands
def raw2tiff(data_path: Path, product_id: str, output_path: Path,
             bands: List[str] = ('B02', 'B03', 'B04', 'B08', 'Cloud')) -> None:

    mission_id, product_level, timestamp1, _, _, tile_id, timestamp2 = product_id.split('_')
    granule_folder = data_path / product_id / 'GRANULE'
    subfolder = [f for f in granule_folder.iterdir() if f.is_dir()][0]
    img_folder, qi_folder = subfolder / 'IMG_DATA', subfolder / 'QI_DATA'

    bands_data = []
    meta = None

    for i, band in enumerate(bands):
        if band in BANDS['10m']:
            band_file = img_folder / 'R10m' / f'{tile_id}_{timestamp1}_{band}_10m.jp2'
            band_res = 10
        elif band == 'Cloud':
            band_file = qi_folder / 'MSK_CLDPRB_20m.jp2'
            band_res = 20

        with rasterio.open(band_file) as src:
            data = src.read(1)  # Read the first band (jp2 usually has one band)

            # Upsample (nearest neighbor interpolation) if not 10 m res
            if band_res > 10:
                m, n = data.shape
                data = cv2.resize(data, (m * (band_res // 10), n * (band_res // 10)),
                                  interpolation=cv2.INTER_NEAREST)

            bands_data.append(data)


        # Copy metadata from the first file
        if meta is None:
            assert band_res == 10  # Copying metadata from a 10 m band
            meta = src.meta.copy()

    # Update metadata for multi-band output
    meta.update({
        "driver": "GTiff",
        "count": len(bands_data),
        "dtype": bands_data[0].dtype
    })

    # Write multi-band GeoTIFF
    out_file = output_path / f'{product_id}.tif'
    with rasterio.open(out_file, "w", **meta) as dst:
        for i, band in enumerate(bands_data, start=1):
            dst.write(band, i)


def cloud_percentage_field(field_folder: Path, s2_timestamp: str, cloud_band: int) -> float:
    field_id = field_folder.stem
    s2_file = field_folder / f'{field_id}_{s2_timestamp}.tif'
    with rasterio.open(s2_file) as src:
        cloud_band = src.read(cloud_band)  # Read the first band (jp2 usually has one band)
    mask_file = field_folder / f'{field_id}_mask.tif'
    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    field_pixels = cloud_band[mask == 1]
    cloud_p = np.sum(field_pixels) / field_pixels.size

    return cloud_p


if __name__ == '__main__':
    data_path = Path('/home/sebastian-hafner/projects/satdes/data/satellite/')
    scene_id = 'S2A_MSIL2A_20240616T100601_N0510_R022_T33UVB_20240616T160951.SAFE'
    raw2tiff(data_path, scene_id, data_path)