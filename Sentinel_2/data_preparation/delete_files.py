from pathlib import Path
import argparse
from tqdm import tqdm
import json

import geopandas as gpd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--deletion_tags', nargs='+', required=True)
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)

    # Loading all fields
    fields = gpd.read_parquet(fields_file)

    deletion_counter = 0

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'
        field_path = dataset_path / 'data' / field_id

        for tag in args.deletion_tags:
            if tag == 'mask':
                file = field_path / f'mask_{field_id}.tif'
                file.unlink()
                deletion_counter += 1
            elif tag == 'polygon':
                file = field_path / f'polygon_{field_id}.parquet'
                file.unlink()
                deletion_counter += 1
            elif tag == 'embeddings':
                file = field_path / f'embeddings_{field_id}.tif'
                file.unlink()
                deletion_counter += 1
            elif tag == 'embeddings_pixelset':
                file = field_path / f'embeddings_pixelset_{field_id}.npy'
                file.unlink()
                deletion_counter += 1
            elif tag == 'bbox':
                file = field_path / f'bbox_{field_id}.parquet'
                file.unlink()
                deletion_counter += 1
            elif tag == 's2':
                s2_metadata_file = field_path / f's2_metadata_{field_id}.json'
                if not s2_metadata_file.exists():
                    continue
                timestamps = json.loads(s2_metadata_file.read_text())
                for t in timestamps['timestamps']:
                    file = field_path / f's2_{field_id}_{t}.tif'
                    if not file.exists():
                        continue
                    file.unlink()
                    deletion_counter += 1
                s2_metadata_file.unlink()
                deletion_counter += 1

    print(f'Deleted {deletion_counter} files')