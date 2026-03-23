from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List

import json
import geopandas as gpd

import random
from data_preparation import phidown_helpers


def generate_metadata(fields_file: Path, data_path: Path, cloud_band: int) -> List[dict]:
    # Fields
    # field_id: Identifier of field
    # crop_id: Crop type identifier
    # harv_t: Harvest date
    # crop_res: Whether there are crop residues
    # s2_t: Acquisition time of S2 image
    # cloud_p: Cloudy pixel percentage

    metadata = []

    # Loading all fields
    assert fields_file.exists() and fields_file.suffix == '.parquet'
    fields = gpd.read_parquet(fields_file)

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = f'field{i}'
        fields_folder = data_path / field_id
        s2_file = [f for f in fields_folder.glob(f'{field_id}_20*.tif') if f.is_file()][0]
        s2_timestamp = s2_file.stem.split('_')[-1]
        cloud_p = phidown_helpers.cloud_percentage_field(fields_folder, s2_timestamp, cloud_band)

        metadata.append({
            'field_id': field_id,
            'crop_id': random.randint(0, 10),
            'harv_t': '2024-08-01',
            'crop_res': random.randint(0, 2),
            's2_t': s2_timestamp,
            'cloud_p': cloud_p,
        })

        if i == 3:
            break

    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', required=True)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
    fields_file, data_path = Path(args.fields_file), Path(args.data_path)
    dataset_metadata = generate_metadata(fields_file, data_path, 5)
    out_file = data_path / 'dataset_metadata.json'
    with open(out_file, 'w') as f:
        json.dump(dataset_metadata, f)
