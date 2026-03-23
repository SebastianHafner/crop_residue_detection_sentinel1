from shapely.geometry import Polygon, box
from shapely.ops import transform
from pyproj import Transformer
from rasterio.transform import Affine
import rasterio
import geopandas as gpd

import numpy as np
from pathlib import Path


def reproject_geometry(geom: Polygon, from_crs:str, to_crs: str) -> Polygon:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return transform(transformer.transform, geom)


# Function to check if pixel is fully contained
def get_fully_contained_mask(geom: Polygon, transform: Affine, height: int, width: int):
    """Only include pixels where all corners are inside the geometry"""
    mask = np.zeros((height, width), dtype="uint8")

    for row in range(height):
        for col in range(width):
            # Get pixel bounds
            left = transform[2] + col * transform[0]
            right = left + transform[0]
            top = transform[5] + row * transform[4]
            bottom = top + transform[4]

            # Create pixel box
            pixel_box = box(left, bottom, right, top)

            # Check if pixel is fully within geometry
            if geom.contains(pixel_box):
                mask[row, col] = 1

    return mask
