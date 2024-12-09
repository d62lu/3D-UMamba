import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetFeaturePropagation, InputEmbedding, PointNetSetAbstractionMsg, PointNetSetAbstraction, MambaBlock, SparseCNN
import torch
import time
import sys

class get_model(nn.Module):
    def __init__(self,num_classes, fps_n_list, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 8 if normal_channel else 0
        self.normal_channel = normal_channel
        embeded_channel = 64

        self.fps_n_list = fps_n_list #[512, 128, 32]

        depth = 1
        drop_path_rate = 0.1
        rms_norm = False
        drop_path = 0.2
        drop_out = 0
        fetch_idx = [0]


        # sconv parameters
        sconv_filter_list = [3, 5, 7]
        sconv_feature_split = [1/4, 1/4, 1/2]
        resolution = 25
        with_se = True
        normalize=True
        
        sconv_feature_list = [int(embeded_channel*sconv_feature_split[i]) for i in range(len(sconv_feature_split))]
        
        # self.sa1 = PointNetSetAbstractionMsg(256, [0.2], [32], in_channel,[[64, 64, 128]])
        # self.global_sa1 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=256, in_channel=128, out_channels=128, layers=1, num_heads=4, head_dim=32)
        # self.sa2 = PointNetSetAbstractionMsg(64, [0.4], [64], 128,[ [128, 128, 256]])
        # self.global_sa2 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=64, in_channel=256, out_channels=256, layers=1, num_heads=8, head_dim=32)
        # self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], True)

        self.embedding = InputEmbedding([0.1], [32], in_channel,[[32, 32, embeded_channel]])

        self.SconvLayers = SparseCNN(in_channel, sconv_feature_list, sconv_filter_list, resolution, with_se, normalize, 0)


        self.sa1 = PointNetSetAbstractionMsg(fps_n_list[0], [0.1, 0.2, 0.4], [32, 64, 128], 64,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.mamba1 = MambaBlock(320, depth, rms_norm, drop_path, fetch_idx, drop_out, drop_path_rate)
        
        self.sa2 = PointNetSetAbstractionMsg(fps_n_list[1], [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.mamba2 = MambaBlock(640, depth, rms_norm, drop_path, fetch_idx, drop_out, drop_path_rate)

        self.sa3 = PointNetSetAbstractionMsg(fps_n_list[2], [0.4, 0.8, 1], [16, 32, 64], 640,[[128, 128, 256], [256, 256, 512], [256, 256, 512]])
        self.mamba3 = MambaBlock(1280, depth, rms_norm, drop_path, fetch_idx, drop_out, drop_path_rate)
        
        self.sa4 = PointNetSetAbstraction(None, None, None, 1280 + 3, [256, 512, 1024], True)

        self.fp4 = PointNetFeaturePropagation(in_channel=1024+320+640+1280+(1280), mlp=[1024, 512])
        
        self.fp3 = PointNetFeaturePropagation(in_channel=512+640, mlp=[512, 256])

        self.fp2 = PointNetFeaturePropagation(in_channel=256+320, mlp=[256, 128])

        self.fp1 = PointNetFeaturePropagation(in_channel=128+3+embeded_channel, mlp=[128, 128])
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gama = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, fps_index_array, series_idx_arrays): #（B C N）， （B 3 N）， （B, 3, 8 N）
        # xyz = xyz.permute(0, 2, 1)  # B C N 
        

        xyz1 = xyz[:, :4, :]
        xyz2 = xyz[:, 7:, :] # 8 channels in total, delete the 3 absolute coordinate channels. raw input channel is 11
        xyz = torch.cat([xyz1,xyz2], dim=1)
        
        
        B, _, N = xyz.shape
        if self.normal_channel:
            norm = xyz
            xyz = xyz[:, :3, :]
        else:
            norm = None
    
        start_time = time.time()
        embeded_points = self.SconvLayers(xyz, norm)
        # xyz, embeded_points = self.embedding(xyz, norm)
        end_time = time.time()
        # print('embedding time:', end_time - start_time)
        
        start_time = time.time()
        l1_xyz, l1_points = self.sa1(xyz, embeded_points, fps_index_array[:,0,:self.fps_n_list[0]]) # fps: B N
        end_time = time.time()
        # print('sa1 time:', end_time - start_time)
        
        start_time = time.time()
        l1_points = self.mamba1(l1_points, series_idx_arrays[:, 0, :, :self.fps_n_list[0]]) # b d n, series B 8 N
        global_feats1 = self.alpha*torch.max(l1_points, 2)[0] # b d
        end_time = time.time()
        # print('mamba1 time:', end_time - start_time)
        
        start_time = time.time()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, fps_index_array[:,1,:self.fps_n_list[1]])
        end_time = time.time()
        # print('sa2 time:', end_time - start_time)
        
        start_time = time.time()
        l2_points = self.mamba2(l2_points, series_idx_arrays[:, 1, :, :self.fps_n_list[1]]) # b d n
        global_feats2 = self.beta*torch.max(l2_points, 2)[0]# b d
        end_time = time.time()
        # print('mamba2 time:', end_time - start_time)
        
        start_time = time.time()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, fps_index_array[:,2,:self.fps_n_list[2]])
        end_time = time.time()
        # print('sa3 time:', end_time - start_time)
        
        start_time = time.time()
        l3_points = self.mamba3(l3_points, series_idx_arrays[:, 2, :, :self.fps_n_list[2]]) # b d n
        global_feats3 = self.gama*torch.max(l3_points, 2)[0]# b d
        end_time = time.time()
        # print('mamba3 time:', end_time - start_time)
        
        start_time = time.time()
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        global_feats4 = l4_points.view(B, 1024)

        x = torch.cat((global_feats1, global_feats2, global_feats3, global_feats4), dim=-1) # 320+640+1024    B D
        x = x.unsqueeze(-1)

        # Feature Propagation layers
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, x)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = self.fp1(xyz, l1_xyz, torch.cat([xyz,embeded_points],1), l1_points)
        end_time = time.time()
        # print('decoder time:', end_time - start_time)
        # sys.stdout.flush()
       
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

class get_loss_weighted(nn.Module):
    def __init__(self):
      super(get_loss_weighted, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.cross_entropy(pred, target, weight)
        #total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss