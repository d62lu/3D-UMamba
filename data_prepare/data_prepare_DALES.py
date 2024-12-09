from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply, read_ply
from helper_tool import DataProcessing as DP

def normalize_scale(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def read_lines(p):
    with open(p, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
        ]
    return lines
    

grid_size = 0.1 # 0.4/0.5 for kitti360
trainset_path = '../data/DALESObjects/train'
testset_path = '../data/DALESObjects/test'
dataset_path = '../data/DALESObjects'

original_pc_folder = dataset_path

train_files = read_lines(os.path.join(trainset_path, "train.txt"))
val_files = read_lines(os.path.join(testset_path, "test.txt"))
print("All files lens: ", len(train_files)+len(val_files))

sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(grid_size))
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

for pc_path in [join(trainset_path, fname) for fname in train_files]:
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    pc = read_ply(pc_path)
    labels = pc['sem_class'].astype(np.uint8)
    xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    # color = np.vstack((pc['red'], pc['green'], pc['blue'])).T.astype(np.uint8)
    intensity = pc['intensity'].astype(np.uint8).reshape(-1,1)
    #  Subsample to save space
    sub_xyz, sub_intensity, sub_labels = DP.grid_sub_sampling(points = xyz, features=intensity, labels=labels, grid_size=grid_size)
    # _, sub_intensity = DP.grid_sub_sampling(xyz, features=intensity, grid_size=grid_size)

    # sub_colors = sub_colors / 255.0
    sub_intensity = sub_intensity / 255.0
    # normalize xyz to unit scale
    # sub_xyz = normalize_scale(sub_xyz)

    sub_ply_file = join(sub_pc_folder, file_name + '_train.txt')
 
    output = np.concatenate((sub_xyz, sub_intensity, sub_labels), axis=-1)
    print('room point cloud:', output.shape)
    #write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    np.savetxt(sub_ply_file, output)

for pc_path in [join(testset_path, fname) for fname in val_files]:
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    pc = read_ply(pc_path)
    labels = pc['sem_class'].astype(np.uint8)
    xyz = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    # color = np.vstack((pc['red'], pc['green'], pc['blue'])).T.astype(np.uint8)
    intensity = pc['intensity'].astype(np.uint8).reshape(-1,1)
    #  Subsample to save space
    sub_xyz, sub_intensity, sub_labels = DP.grid_sub_sampling(points = xyz, features=intensity, labels=labels, grid_size=grid_size)
    # _, sub_intensity = DP.grid_sub_sampling(xyz, features=intensity, grid_size=grid_size)

    # sub_colors = sub_colors / 255.0
    sub_intensity = sub_intensity / 255.0
    # normalize xyz to unit scale
    # sub_xyz = normalize_scale(sub_xyz)

    sub_ply_file = join(sub_pc_folder, file_name + '_test.txt')
 
    output = np.concatenate((sub_xyz, sub_intensity, sub_labels), axis=-1)
    print('room point cloud:', output.shape)
    #write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    np.savetxt(sub_ply_file, output)




