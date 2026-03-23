from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

import geopandas as gpd
import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='ee-sebastianhafner')
JOB_COUNTER = 0

SPECTRAL_BANDS = ['B2', 'B3', 'B4', 'B8']
CLOUD_FILTER = 60
CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def add_cloud_bands(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


def add_shadow_bands(img):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cld_shdw_mask(img):
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER * 2 / 20)
                   .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
                   .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)


def apply_cld_shdw_mask(img):
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def add_field_cloud_coverage(aoi):
    """Returns a function that computes cloud coverage percentage over the field AOI"""

    def add_field_cloud_coverage(aoi):
        """Returns a function that computes cloud coverage percentage over the field AOI"""

        def compute_coverage(img):
            # Add cloud/shadow mask bands
            img_with_mask = add_cld_shdw_mask(img)

            # Get the cloudmask band (1 = cloud/shadow, 0 = clear)
            cloud_mask = img_with_mask.select('cloudmask')

            # Calculate mean over the field AOI (gives fraction 0-1)
            coverage_stats = cloud_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=20,  # Use 20m to match cloud mask resolution
                maxPixels=1e9
            )

            coverage_percentage = ee.Number(coverage_stats.get('cloudmask')).multiply(100)

            return img_with_mask.set('field_cloud_coverage', coverage_percentage)

        return compute_coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--start_index', type=int, required=False, default=0)
    parser.add_argument('--cloud_threshold', type=int, required=False, default=20)
    parser.add_argument('--crs', type=str, required=False, default='EPSG:3006')
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)

    # Loading all fields
    fields = gpd.read_parquet(fields_file)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        if i < args.start_index:
            continue
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'
        field_path = dataset_path / 'data' / field_id

        # Loading bbox
        gdf_bbox = gpd.read_parquet(field_path / f'bbox_{field_id}.parquet').to_crs('EPSG:4326')
        bbox = gdf_bbox.geometry.iloc[0]
        bounds = bbox.bounds
        ee_bbox = ee.Geometry.BBox(bounds[0], bounds[1], bounds[2], bounds[3])
        ee_bbox = ee_bbox.buffer(-5)

        # Get S2 collection with cloud probability
        s2_sr_cld_col = get_s2_sr_cld_col(ee_bbox, args.start_date, args.end_date)

        # Add field cloud coverage property AND apply masking to each image
        s2_masked_with_coverage = s2_sr_cld_col.map(add_field_cloud_coverage(ee_bbox))

        # Filter by cloud coverage threshold
        s2_filtered = s2_masked_with_coverage.filter(
            ee.Filter.lt('field_cloud_coverage', args.cloud_threshold)
        )

        # Convert to list to iterate
        s2_list = s2_filtered.toList(s2_filtered.size())
        n_scenes = s2_list.size().getInfo()
        if JOB_COUNTER + n_scenes > 3_000:
            print(f'Stopped at field {field_id} ({i}) due to too many concurrent jobs')
            break

        if n_scenes == 0:
            print("No scenes meet the cloud coverage threshold. Skipping field.")
            continue

        # Store metadata
        timestamps = []
        cloud_coverages = []

        # Export each scene individually
        for idx in range(n_scenes):
            # Get the already-masked image from the list
            image = ee.Image(s2_list.get(idx))

            # Get properties
            image_info = image.getInfo()
            system_index = image_info['properties']['system:index']
            date_acquired = image_info['properties']['system:time_start']
            cloud_coverage = image_info['properties']['field_cloud_coverage']

            date_str = datetime.fromtimestamp(date_acquired / 1000).strftime('%Y%m%d')

            # Store metadata
            timestamps.append(date_str)
            cloud_coverages.append(cloud_coverage)

            # Create export task (image is already masked)
            task = ee.batch.Export.image.toDrive(
                image=image.select(SPECTRAL_BANDS),
                description=f'{field_id}_{date_str}',
                scale=10,
                region=ee_bbox,
                fileNamePrefix=f's2_{field_id}_{date_str}',
                folder='satdes_dataset',
                crs=args.crs,
                fileFormat='GeoTIFF',
                maxPixels=1e9
            )
            # task.start()
            JOB_COUNTER += 1

        # Save metadata
        field_metadata = {
            'field_id': field_id,
            'n_scenes': n_scenes,
            'timestamps': timestamps,
            'cloud_coverages': cloud_coverages,
            'cloud_threshold': args.cloud_threshold
        }

        with open(field_path / f's2_metadata_{field_id}.json', 'w') as f:
            json.dump(field_metadata, f, indent=2)