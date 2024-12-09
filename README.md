# 3D-UMamba
3D-UMamba: 3D U-Net with state space model for semantic segmentation of multi-source LiDAR point clouds

This is a Pytorch implementation of 3D-UMamba.

## Abtract

Segmentation of point clouds is foundational to numerous remote sensing applications. Recently, the development of Transformers has further improved segmentation techniques thanks to their great long-range context modeling capability. However, Transformers have quadratic complexity in inference time and memory, which both limits the input size and poses a strict hardware requirement. This paper presents a novel 3D-UMamba network with linear complexity, which is the earliest to introduce the Selective State Space Model (i.e., Mamba) to multi-source LiDAR point cloud processing. 3D-UMamba integrates Mamba into the classic U-Net architecture, presenting outstanding global context modeling with high efficiency and achieving an effective combination of local and global information. In addition, we propose a simple yet efficient 3D-token serialization approach (Voxel-based Token Serialization, i.e., VTS) for Mamba, where the Bi-Scanning strategy enables the model to collect features from all input points in different directions effectively. The performance of 3D-UMamba on three challenging LiDAR point cloud datasets (airborne MultiSpectral LiDAR (MS-LiDAR), aerial DALES, and vehicle-mounted Toronto-3D) demonstrated its superiority in multi-source LiDAR point cloud semantic segmentation, as well as the strong adaptability of Mamba to different types of LiDAR data, exceeding current state-of-the-art models. Ablation studies demonstrated the higher efficiency and lower memory costs of 3D-UMamba than its Transformer-based counterparts.


## Architecture

<img width="580" alt="1733765091194" src="https://github.com/user-attachments/assets/cee06ef3-7db0-40fe-b7bf-7df0fed2a27d">

## Install
The latest codes are tested on CUDA11.3 and above, PyTorch 1.10.1 and Python 3.9.
For mamba installation, please refer to PointMamba (https://github.com/LMD0311/PointMamba)



## Data Preparation
Download DAELS (https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) and save in "data/".

## Run
1. Data preprocessing
   Folder: data_prepare
   ```
  python data_prepare_DALES.py
  python DALES_sample_generation
  ```

2. Model Training
  ```
  python train_DALES.py --model mamba_msg --log_dir 3dumamba --learning_rate 0.01 --batch_size 4 --optimizer SGD --epoch 120 --gpu 0 --npoint 8192 --weighted_loss True
  ```

## Acknowledgement

We would like to express our sincere gratitude to PointMamba (https://github.com/LMD0311/PointMamba) and PVCNN (https://github.com/mit-han-lab/pvcnn/tree/master)  for their valuable work on, which has significantly contributed to the development of this project.
