# coding=gb2312
import math
import numpy as np
import glob
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


dataset_path = '../data/DALESObjects/input_0.100/'
file_paths=dataset_path+'*.txt' # locate the files whose names include 'Site'
file_paths=glob.glob(file_paths) # collecting all 'Site**_***' files to the list

def norm_and_surface_variance(pc):
    points = pc[:,:-1]
    labels = pc[:,-1:]
    point_cloud = pc[:,4:7]

    pca = PCA(n_components=3)


    neighbor_radius = 1.0


    nbrs = NearestNeighbors(radius=neighbor_radius, algorithm='ball_tree').fit(point_cloud)


    distances, indices = nbrs.radius_neighbors(point_cloud)

    local_features = []
    norm_features = []
    for i in range(len(point_cloud)):
  
        neighbor_indices = indices[i]
        if len(neighbor_indices) <= 2:
            local_features.append([1, 0, 0])  
            norm_features.append([0, 0, 0])
        else:

            neighborhood = point_cloud[neighbor_indices]



            pca.fit(neighborhood)
            local_features.append(pca.explained_variance_ratio_)
            norm_features.append(pca.components_[2])

    local_features = np.array(local_features)
    norm_features = np.array(norm_features)
    # print(np.sum(local_features, axis=-1).shape)
    surface_variance = (local_features[:,2]/(np.sum(local_features, axis=-1))).reshape(-1, 1)
    # print(surface_variance.shape)
    output_point_cloud = np.concatenate([points, norm_features, surface_variance, labels], axis = -1) # rx, ry, rz, intensity, ax, ay, az, nx, ny, nz, sv, l
    return output_point_cloud


def coordinate_normalize(input_data):
        input_xyz = input_data[:, :, :3]
        max_xyz = np.max(input_xyz, axis=1)
        min_xyz = np.min(input_xyz, axis=1)
        mean_xyz = (max_xyz + min_xyz) / 2
        mean_xyz = np.expand_dims(mean_xyz, axis=1)
        div = np.max(input_xyz - mean_xyz, axis=1)
        max = np.max(div, axis=-1)
        div = np.expand_dims(div, axis=1)
        max = np.expand_dims(max, axis=1)
        max = np.expand_dims(max, axis=1)
        new_xyz = (input_xyz - mean_xyz) / max
        input_data[:, :, :3] = new_xyz
        return input_data

block_size_list = [20]


     
    
for file_path in tqdm(file_paths):
    file_name = os.path.basename(file_path)[:-4]
    #print(file_name)

    

    points = np.loadtxt(file_path)
    #print('raw shape:', points.shape)
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    x = coord_max[0] - coord_min[0]
    y = coord_max[1] - coord_min[1]
    # print(x)
    # print(y)
    
    for size_ in range(len(block_size_list)):
        block_size = block_size_list[size_]
        min_npoints = 4096
        
        data_folder = dataset_path + 'Block_s' + str(block_size)+'_min' + str(min_npoints) +'_norm_enhance/' # bulid folder if not exist
        data_root = data_folder
        if not os.path.exists(data_folder):
                os.makedirs(data_folder)
        
        center_x = []
        center_y = []
        if x<block_size:
            center_nx = 1
            center_x.append(x/2)
        else:
            center_nx = math.ceil(x/block_size)
            center_x.append(block_size/2)
            center_x.append(x-block_size/2)
            center_x_length = x-block_size
    
            for i in range(1, center_nx-1):
                center_x.append(block_size/2 + center_x_length/(center_nx-1)*i)
        # print(center_nx)
        # center_x.append(0)
        center_x = sorted(center_x) + coord_min[0]
        # print(center_x)
    
        if y<block_size:
            center_ny = 1
            center_y.append(y/2)
        else:
            center_ny = math.ceil(y/block_size)
            center_y.append(block_size/2)
            center_y.append(y-block_size/2)
            center_y_length = y-block_size
    
            for i in range(1, center_ny-1):
                center_y.append(block_size/2 + center_y_length/(center_ny-1)*i)
        # print(center_ny)
        # center_x.append(0)
        center_y = sorted(center_y)+ coord_min[1]
        # print(center_y)
    
        coordinates = [[xi, yi] for xi in center_x for yi in center_y]
        
        data_npy = []
    
        for i in range(len(coordinates)):
            # print('block idx:', str(i))
            block_x_min = coordinates[i][0] - block_size/2
            block_x_max = coordinates[i][0] + block_size/2
            block_y_min = coordinates[i][1] - block_size/2
            block_y_max = coordinates[i][1] + block_size/2
            point_idxs = np.where((points[:, 0] >= block_x_min) & (points[:, 0] <= block_x_max) & (points[:, 1] >= block_y_min) & (points[:, 1] <= block_y_max))[0]
            
            if point_idxs.size == 0:
                print('0 Existing!!!')
                continue
                
            if point_idxs.size >= min_npoints:
                selected_point_idxs = np.random.choice(point_idxs.size, min_npoints, replace=False)
            else:
                selected_point_idxs = np.random.choice(point_idxs.size, min_npoints, replace=True)
            
            selected_point_idxs = point_idxs[selected_point_idxs]
            selected_points = points[selected_point_idxs, :]
            
            current_points = np.zeros((selected_point_idxs.size , 8))  # num_point * 5
            
            current_points[:, :4] = selected_points[:, :4]
            current_points[:, 4:7] = selected_points[:, :3]
            current_points[:, 7] = selected_points[:, 4]
            
            current_points = norm_and_surface_variance(current_points)
    
            current_points = np.expand_dims(current_points, axis=0)
            current_points = coordinate_normalize(current_points) # nx, ny, nz, intensity, ax, ay, az, label # rx ry rz intensity ax ay az label
            #print('sample shape:', current_points.shape)
            data_npy.append(current_points)
        file_save_path = data_root + file_name + '.npy'
        data_all = np.concatenate(data_npy, axis = 0)
        np.save(file_save_path, data_all)




