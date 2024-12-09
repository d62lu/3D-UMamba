# 3D-UMamba
3D-UMamba: 3D U-Net with state space model for semantic segmentation of multi-source LiDAR point clouds

This is a Pytorch implementation of 3D-UMamba.

## Abtract

Segmentation of point clouds is foundational to numerous remote sensing applications. Recently, the development of Transformers has further improved segmentation techniques thanks to their great long-range context modeling capability. However, Transformers have quadratic complexity in inference time and memory, which both limits the input size and poses a strict hardware requirement. This paper presents a novel 3D-UMamba network with linear complexity, which is the earliest to introduce the Selective State Space Model (i.e., Mamba) to multi-source LiDAR point cloud processing. 3D-UMamba integrates Mamba into the classic U-Net architecture, presenting outstanding global context modeling with high efficiency and achieving an effective combination of local and global information. In addition, we propose a simple yet efficient 3D-token serialization approach (Voxel-based Token Serialization, i.e., VTS) for Mamba, where the Bi-Scanning strategy enables the model to collect features from all input points in different directions effectively. The performance of 3D-UMamba on three challenging LiDAR point cloud datasets (airborne MultiSpectral LiDAR (MS-LiDAR), aerial DALES, and vehicle-mounted Toronto-3D) demonstrated its superiority in multi-source LiDAR point cloud semantic segmentation, as well as the strong adaptability of Mamba to different types of LiDAR data, exceeding current state-of-the-art models. Ablation studies demonstrated the higher efficiency and lower memory costs of 3D-UMamba than its Transformer-based counterparts.


## Architecture

<img width="580" alt="1733765091194" src="https://github.com/user-attachments/assets/cee06ef3-7db0-40fe-b7bf-7df0fed2a27d">

## Install
The latest codes are tested on CUDA10.1, PyTorch 1.6 and Python 3.8.

## Data Preparation
Download alignment ModelNet (https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in "data/modelnet40_normal_resampled/".

## Run
```
python train_classification.py --use_normals --model pointnet2_cls_msg --log_dir pointnet2_cls_msg_github --learning_rate 0.01 --batch_size 16 --optimizer SGD --epoch 300 --process_data
```

