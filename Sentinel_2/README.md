

## Setup

## Data Download


### Embeddings

1. Download all embeddings tiles intersecting with the fields boundaries:

```console
python embeddings_download.py --fields_file *field boundaries* --out_path *output path*
```

2. Creating dataset by cropping embedding tiles to field boundaries:

```console
python embeddings_field_extraction.py --fields_file *field boundaries* --data_path *path to embeddings*
```

The directory containing the embeddings will have the following structure:

```
*output/data path*
│   embeddings_registry.parquet
│   dataset_metadata.json
│
└───global_0.1_degree_representation
│   │
│   └───2024
│       │ 
│       └───grid_13.05_55.55
│       │   │
│       │   └───grid_13.05_55.55_2024.tiff
│       │
│       └───grid_13.85_55.85 
│       │   ...
│   
└───fields
│   │
│   └───field0
│   │   │ field0.parquet  # Field boundary
│   │   │ field0_embeddings.tif  # Embeddings cropped to field
│   │   │ field0_mask.tif  # Binary mask of field pixels  
│   │
│   └───field1
│   │   ...
```

### Sentinel-2


1. Download all embeddings tiles intersecting with the fields boundaries:

```console
python embeddings_download.py --fields_file *field boundaries* --out_path *output path*
```

2. Creating dataset by cropping embedding tiles to field boundaries:

```console
python embeddings_field_extraction.py --fields_file *field boundaries* --data_path *path to embeddings*
```

The directory containing the embeddings will have the following structure:

```
*output/data path*
│   embeddings_metadata.parquet  # Metadata
│
└───global_0.1_degree_representation
│   │
│   └───2024
│       │ 
│       └───grid_13.05_55.55
│       │   │
│       │   └───grid_13.05_55.55_2024.tiff
│       │
│       └───grid_13.85_55.85 
│       │   ...
│   
└───fields
│   │
│   └───field0
│   │   │ field0.parquet  # Field boundary
│   │   │ field0_embeddings.tif  # Embeddings cropped to field
│   │   │ field0_mask.tif  # Binary mask of field pixels  
│   │
│   └───field1
│   │   ...
```



## Model Training


## Deployment

