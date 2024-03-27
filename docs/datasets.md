## Datasets

This project uses several datasets and files organized in a specific structure. Here's an overview of the datasets and their purposes:

```bash
├──datasets
│   ├── scannet
│   │   ├── scans # scannet scans
│   │   │   ├── scene0000_00 # only these 4 files are used.
│   │   │   │   ├── scene0000_00.txt
│   │   │   │   ├── scene0000_00_vh_clean_2.ply
│   │   │   │   ├── scene0000_00.aggregation.json
│   │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
│   │   │   ├── ......
│   │   │   ├── scene0706_00
│   │   ├── scan2cad # scan2cad, only the following 1 file is used.
│   │   │   ├── full_annotations.json
│   │   ├── scannetv2-labels-combined.tsv # scannet label mappings
│   │   ├── processed_data # preprocessed data
│   │   │   ├── scene0000_00 
│   │   │   │   ├── bbox.pkl
│   │   │   │   ├── data.npz
│   │   │   ├── ......
│   │   │   ├── scene0706_00
│   │   ├── rfs_label_map.csv # generated label mappings
│   ├── ShapeNetCore.v2 # shapenet core v2 dataset
│   │   ├── 02954340
│   │   ├── ......
│   │   ├── 04554684
│   ├── ShapeNetv2_data # preprocessed shapenet dataset
│   │   ├── watertight_scaled_simplified
│   ├── bsp # the pretrained bsp model
│   │   ├── zs
│   │   ├── database_scannet.npz
│   │   ├── model.pth
│   ├── splits # data splits
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
```

### Scannet

Scannet is a richly-annotated dataset of indoor scenes. It contains RGB-D scans of scene interiors.

- `scans`: This directory contains the Scannet scans. Each scene (e.g., `scene0000_00`) has four associated files:
    - `.txt`: Contains metadata about the scan.
    - `_vh_clean_2.ply`: The 3D point cloud data.
    - `.aggregation.json`: Contains object segmentation information.
    - `.segs.json`: Contains semantic segmentation information.
- `scan2cad`: This directory contains the Scan2CAD dataset, which aligns CAD models with Scannet scans. The `full_annotations.json` file is used from this dataset.
- `scannetv2-labels-combined.tsv`: This file maps Scannet labels to NYU40 labels.
- `processed_data`: This directory contains preprocessed data for each scene. Each scene directory (e.g., `scene0000_00`) contains:
    - `bbox.pkl`: Bounding box data for the scene.
    - `data.npz`: Preprocessed data for the scene.
- `rfs_label_map.csv`: This file contains generated label mappings.

### ShapeNetCore.v2

ShapeNetCore.v2 is a dataset of 3D CAD models from ShapeNet. It is organized by category (e.g., `02954340`, `04554684`).

### ShapeNetv2_data

This directory contains preprocessed ShapeNet dataset.

- `watertight_scaled_simplified`: This directory contains watertight and scaled versions of the ShapeNet models.

### BSP

This directory contains the pretrained BSP (Binary Space Partitioning) model.

- `zs`: This directory contains the zero-shot shape generation results.
- `database_scannet.npz`: This file contains the Scannet database for the BSP model.
- `model.pth`: This is the pretrained model file.

### Scan2CAD

`Scan2CAD` is a dataset that aligns 3D CAD models with 3D scans. It contains keypoint correspondences, objects, and scans. The `full_annotations.json` file from this dataset is used in the project. This dataset is particularly useful for tasks related to 3D object pose annotation, providing precise spatial information about the objects in the scans.

### Splits

This directory contains the data splits for training, validation, and testing (`train.txt`, `val.txt`, `test.txt`).
