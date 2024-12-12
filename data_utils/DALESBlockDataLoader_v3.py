import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def fps_series_func(points, voxel_indices, samplepoints_list):
    pad_width = points.shape[0]
    points = torch.Tensor(points).float().cuda().unsqueeze(0)
    voxel_indices = torch.Tensor(voxel_indices).float().cuda().unsqueeze(0)
    fps_index_list = []
    series_idx_lists = []

    x1y1z1 = [1, 1, 1]
    x0y1z1 = [-1, 1, 1]
    x1y0z1 = [1, -1, 1]
    x0y0z1 = [-1, -1, 1]
    x1y1z0 = [1, 1, -1]
    x0y1z0 = [-1, 1, -1]
    x1y0z0 = [1, -1, -1]
    x0y0z0 = [-1, -1, -1]

    series_list = []
    #series_list.append(x1y1z1)
    #series_list.append(x0y1z1)
    #series_list.append(x1y0z1)
    series_list.append(x0y0z1)
    series_list.append(x1y1z0)
    #series_list.append(x0y1z0)
    #series_list.append(x1y0z0)
    #series_list.append(x0y0z0)

    for i in range(len(samplepoints_list)):
        S = samplepoints_list[i]
        xyz = points[:, :,:3]

        fps_index=farthest_point_sample(xyz, S)

        points = index_points(points, fps_index)
        new_voxel_indices = index_points(voxel_indices, fps_index).squeeze(0).cpu().data.numpy()
        voxel_indices = index_points(voxel_indices, fps_index)

        fps_index=fps_index.cpu().data.numpy()
        padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(padded_fps_index)
        


        series_idx_list = []
        for j in range(len(series_list)):
            
            series = series_list[j]
            new_voxel_indices_ForSeries = new_voxel_indices*series
            sorting_indices = np.expand_dims(np.lexsort((new_voxel_indices_ForSeries[:, 0], new_voxel_indices_ForSeries[:, 1], new_voxel_indices_ForSeries[:, 2])), axis=0)
            padded_sorting_indices = np.expand_dims(np.pad(sorting_indices, ((0, 0), (0, pad_width - sorting_indices.shape[1])), mode='constant'), axis=0)
            series_idx_list.append(padded_sorting_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1) # 1 8 N (padding 0)_
        series_idx_lists.append(series_idx_array)

    series_idx_arrays = np.concatenate(series_idx_lists, axis=0) # 3 8 N 
    fps_index_array = np.vstack(fps_index_list) # 3 N (padding 0)_

    return fps_index_array, series_idx_arrays

def voxelization(points, voxel_size):
        """
        Perform voxelization on a given point cloud.
        
        Parameters:
        points (numpy.ndarray): Nx3 array of points (x, y, z).
        voxel_size (float): Size of the voxel grid.
        
        Returns:
        numpy.ndarray: Nx3 array of voxelized coordinates.
        """
        # Calculate the voxel indices
        voxel_indices = np.floor(points[:,:3] / voxel_size).astype(np.int32)

        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]

        bounding_box = coord_max - coord_min

        voxel_total = np.ceil(bounding_box[0]*bounding_box[1]*bounding_box[2] / voxel_size**3).astype(np.int32) # 25*25*25
        voxel_valid = np.unique(voxel_indices, axis=0)

        
        return points, voxel_indices, voxel_total, voxel_valid

class DALESDataset(Dataset):
    def __init__(self, split='train', data_root='../data/rs_data/', fps_n_list = [512, 128, 32], label_number = 8, npoints = 8192):
        super().__init__() 

        self.fps_n_list = fps_n_list
        self.npoints = npoints
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if (str(npoints) + '_clean.npy') in room] # 
        #test_list = ['5135_54435', '5080_54470', '5120_54445', '5155_54335',  '5175_54395']
        test_list = ['5135_54435']
        if split == 'train':
            rooms_split = [room for room in rooms if 'Train' in room] #_5100
        else:
            #rooms_split = [room for room in rooms if any(item in room for item in test_list)] #_test
            rooms_split = [room for room in rooms if 'Test' in room] #_test _5135

        self.sample_points, self.sample_labels = [], []
        self.fps_index_array_list, self.series_idx_arrays_list = [], []
        labelweights = np.zeros(label_number)
        voxel_size = 0.4
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N,4096,8
            #print('room_data shape:', room_data.shape)
            for i in tqdm(range(room_data.shape[0])):
                points, labels = room_data[i][:, :-1], room_data[i][:, -1]  #N,4096,7; N,4096,1 
                labels = labels - 1



                #print(labels)
                tmp, _ = np.histogram(labels, range(label_number+1))
                #print(tmp)
                labelweights += tmp
                
                coor_min = np.amin(points, axis=0)[:3]
                points[:,2] = points[:,2] - coor_min[2]


                array = np.arange(points.shape[0])
                np.random.shuffle(array)

                points = points[array]
                labels = labels[array]
                
                points, voxel_indices, voxel_total, voxel_valid = voxelization(points, voxel_size)
                fps_index_array, series_idx_arrays = fps_series_func(points, voxel_indices, self.fps_n_list) # (3, N) 和 （3, 8, N）。3：三层降采样，前面的N是降采样序列，后面的N是排序序列。8是有8个方向的排序
        
    
                self.sample_points.append(points), self.sample_labels.append(labels) #4096,6; 4096,1
                self.fps_index_array_list.append(fps_index_array), self.series_idx_arrays_list.append(series_idx_arrays) #4096,6; 4096,1
            

        # self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.sample_points), split))
        
        self.labelweights = np.ones(label_number)
        if split == 'train':
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)


    
    def __getitem__(self, idx):
        
        points = self.sample_points[idx]   # 4096 * 6/7
        labels = self.sample_labels[idx]   # 4096 * 1
        fps_index_array = self.fps_index_array_list[idx]
        series_idx_arrays = self.series_idx_arrays_list[idx] 


        
        return points, labels, fps_index_array, series_idx_arrays

    def __len__(self):
        return len(self.sample_points)

class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(13)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(14))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    # data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    # num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = RSDataset(split='train')
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()