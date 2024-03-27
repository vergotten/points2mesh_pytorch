## Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. SQL Libraries and Database Management(#sql-libraries-and-database-management)
    - [Docker Setup](#docker-setup)
    - [Ubuntu Environment Setup with CUDA and cuDNN](#ubuntu-environment-setup-with-cuda-and-cudnn)
4. [Preparing the Data](#preparing-the-data)
5. [Training](#training)
6. [Testing](#testing)
7. [Evaluation](#evaluation)

## Overview

This project is a comprehensive framework for 3D object recognition, annotation, and reconstruction from point clouds. It leverages advanced neural network architectures including Binary Space Partitioning (BSP) tree-based networks, U-Net-based networks, Transformer networks, and PointNet++ networks.

The BSP and U-Net-based networks enable efficient and accurate 3D reconstruction from point clouds. Transformer networks, renowned for their effectiveness in natural language processing tasks, are employed to handle the complex relationships and structures within 3D data. PointNet++, a highly effective network for point cloud processing, is used to capture both fine details and global structures.

In addition, the project utilizes CUDA libraries, specifically `spconv`, to efficiently handle sparse tensors. This ensures that the project can handle large amounts of data and complex computations, making it suitable for a wide range of applications.

## SQL Libraries and Database Management

This project also serves as a robust framework for handling databases using all major SQL libraries. It provides functionalities for collecting, managing, and visualizing point data for further use in comparison and analysis. The project leverages SQL commands and Python libraries for SQL databases to efficiently manage and manipulate 3D object data, enhancing the project's capabilities in handling complex 3D reconstructions.

This combination of advanced neural network architectures, efficient data handling techniques, and comprehensive database management makes this project a truly badass framework for anyone looking to dive deep into the world of 3D data, machine learning, and database management.

---

## Environment Setup

### Docker Setup

Instructions on how to build and run the Docker image.

```bash
# Build Docker image
docker build -t pcd2mesh:latest .

# Run Docker container with GPU support
docker run --gpus all -it --rm -v $(pwd):/app pcd2mesh:latest bash
```

### Ubuntu Environment Setup with CUDA and cuDNN

This project contains a shell script for setting up an Ubuntu environment with CUDA and cuDNN. The setup has been tested on Ubuntu 16.04 (20.04) with CUDA 9.0 (10.0), cuDNN 7.3, and Python 3.7.


## Preparing the Data

The first step in the process is to prepare the data. This involves downloading the preprocessed data and label map from the provided links and placing them under `./datasets/scannet`. The preprocessed data is approximately 3.3GB in size.

If you wish to process the data yourself, you will need to download the [ScanNet](http://www.scan-net.org/), [Scan2CAD](https://github.com/skanti/Scan2CAD), and [ShapeNet](https://shapenet.org/) datasets and place them in the appropriate locations.

To preprocess the ScanNet data, run the following command:

```bash
python data/generate_data_relabel.py
```

This command launches 16 processes by default to speed up processing. The process should complete in about 10 minutes. Despite the potential mismatch between instance labels and CAD annotations, which may result in some warnings, the process will not be affected. The command generates the label map `./datasets/scannet/rfs_label_map.csv` and saves `data.npz, bbox.pkl` for each scene under `./datasets/scannet/processed_data/`.

Next, download the preprocessed ShapeNet (simplified watertight mesh) following the instructions provided by [RfDNet](https://github.com/yinyunie/RfDNet) into `ShapeNetv2_data`. Only the `watertight_scaled_simplified` is used for mesh retrieval and evaluation.

Finally, download the pretrained BSP-Net checkpoint and extracted GT latent codes from the provided link. If you wish to generate them yourself, please check the [BSP_CVAE repository](https://github.com/ashawkey/bsp_cvae) to generate the ground truth latent shape codes (`zs` folder), the pretrained model (`model.pth`), and the assistant code database (`database_scannet.npz`).

## Training

The training process is divided into two phases: point-wise and proposal-wise. To train the model, use the following commands:

```bash
# train phase 1 (point-wise)
python train.py --config config/rfs_phase1_scannet.yaml

# train phase 2 (proposal-wise)
python train.py --config config/rfs_phase2_scannet.yaml
```

Please check the config files for more options.

## Testing

To generate completed instance meshes, use the following commands:

```bash
# test after training phase 2
python test.py --config config/rfs_phase2_scannet.yaml
# example path for the meshes: ./exp/scannetv2/rfs/rfs_phase2_scannet/result/epoch256_nmst0.3_scoret0.05_npointt100/val/trimeshes/

# test with a specified checkpoint
python test.py --config config/rfs_pretrained_scannet.yaml --pretrain ./pointgroup_phase2_scannet-000000256.pth
```

A pretrained model is provided at the given link.

To visualize the intermediate point-wise results, use the following commands:

```bash
python util/visualize.py --task semantic_gt --room_name all
python util/visualize.py --task instance_gt --room_name all
# after running test.py, may need to change `--result_root` to the output directory, check the script for more details.
python util/visualize.py --task semantic_pred --room_name all
python util/visualize.py --task instance_pred --room_name all
```

## Evaluation

Four metrics are provided for evaluating the quality of the instance mesh reconstruction. For the IoU evaluation, [binvox](https://www.patrickmin.com/binvox/) is used to voxelize meshes (via trimesh's API), so ensure it can be found in the system path.

First, prepare the GT instance meshes by running the following command:

```bash
python data/scannetv2_inst.py # prepare at "./datasets/gt_meshes"
```

Assuming the generated meshes are under "./pred_meshes", you can evaluate them using the following commands:

```bash
# IoU
python evaluation/iou/eval.py ./datasets/gt_meshes ./pred_meshes

# CD
python evaluation/cd/eval.py ./datasets/gt_meshes ./pred_meshes

# LFD
python evaluation/lfd/eval.py ./datasets/gt_meshes ./pred_meshes

# PCR
python evaluation/pcr/eval.py ./pred_meshes
```
