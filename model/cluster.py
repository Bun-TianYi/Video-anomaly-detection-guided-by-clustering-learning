import sys
import os
# from datasets_sequence import multi_train_datasets, multi_test_datasets
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from einops import rearrange


def cluster_alpha(max_n=40):
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    # max_n = 40  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
    alphas = alphas / constant_value
    # print(alphas)
    return alphas


class PosSoftAssign(nn.Module):
    def __init__(self, dims=1, alpha=1.0):
        super(PosSoftAssign, self).__init__()
        self.dims = dims
        self.alpha = alpha

    def forward(self, x, alpha=None):
        if not alpha == None:
            self.alpha = alpha
        x_max, _ = torch.max(x, self.dims, keepdim=True)
        exp_x = torch.exp(self.alpha * (x - x_max))
        soft_x = exp_x / (exp_x.sum(self.dims, keepdim=True))
        return soft_x


class NegSoftAssign(nn.Module):
    def __init__(self, dims=1, alpha=32.0):
        super(NegSoftAssign, self).__init__()
        self.dims = dims
        self.alpha = alpha

    def forward(self, x, alpha=None):
        if not alpha == None:
            self.alpha = alpha

        x_min, _ = torch.min(x, self.dims, keepdim=True)
        exp_x = torch.exp((-self.alpha) * (x - x_min))  # 这是一个类似于紧缩的方法
        soft_x = exp_x / (exp_x.sum(self.dims, keepdim=True))
        return soft_x


class EuclidDistance_Assign_Module(nn.Module):
    def __init__(self, feature_dim, cluster_num=256, maxpool=1, soft_assign_alpha=32.0):
        super(EuclidDistance_Assign_Module, self).__init__()
        self.euclid_dis = torch.cdist
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = nn.LayerNorm(feature_dim)

        self.assign_func = NegSoftAssign(-1, soft_assign_alpha)
        self.register_param()

    def register_param(self, ):
        cluster_center = nn.Parameter(torch.rand(self.cluster_num, self.feature_dim), requires_grad=True)
        identity_matrix = nn.Parameter(torch.eye(self.cluster_num), requires_grad=False)
        self.register_parameter('cluster_center', cluster_center)
        self.register_parameter('identity_matrix', identity_matrix)
        return

    def self_similarity(self):

        return self.euclid_dis(self.cluster_center, self.cluster_center)

    def forward(self, x, alpha=None):
        #   传入x尺寸为B*D*H*W*C
        x_temp = x.clone()
        x = self.norm(x_temp)
        B, D, H, W, C = x.shape
        x_re = rearrange(x, 'B D H W C -> B (D H W) C')
        soft_assign = self.euclid_dis(x_re, self.cluster_center.unsqueeze(0))  # 这里返回的是向量间的距离
        feature_label = torch.argmin(soft_assign, dim=2, keepdim=True)
        feature_label = rearrange(feature_label, 'B N L -> (B N L)')
        # 若是输入b*w*h*c的矩阵，返回则是b*w*h*1
        x_distance = rearrange(soft_assign, 'B (D H W) CN -> B D H W CN', D=D, H=H, W=W)
        x_distance_assign = self.assign_func(x_distance, alpha)
        cluster_dist = self.self_similarity()
        kk = self.cluster_center.clone()
        x_rec = x_distance_assign @ kk
        feature = rearrange(x_re, 'B DHW C -> (B DHW) C')


        return x_distance, x_distance_assign, cluster_dist, x_rec, feature, feature_label


class Space_EuclidDistance_Assign_Module(nn.Module):
    def __init__(self, feature_dim, cluster_num=128, space_size=28, maxpool=1, soft_assign_alpha=32.0):
        super(Space_EuclidDistance_Assign_Module, self).__init__()
        self.euclid_dis = torch.cdist
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = nn.LayerNorm(feature_dim)
        self.space_size = space_size * space_size
        self.assign_func = NegSoftAssign(-1, soft_assign_alpha)
        self.register_param()

    def register_param(self, ):
        cluster_center = nn.Parameter(torch.rand(self.feature_dim, self.cluster_num, self.space_size), requires_grad=True)
        ident_temp = torch.empty((self.feature_dim, self.cluster_num, self.cluster_num))
        for i in range(self.feature_dim):
            ident_temp[i] = torch.eye(self.cluster_num)
        identity_matrix = nn.Parameter(ident_temp, requires_grad=False)
        self.register_parameter('cluster_center', cluster_center)
        self.register_parameter('identity_matrix', identity_matrix)
        return

    def self_similarity(self):
        return self.euclid_dis(self.cluster_center, self.cluster_center)

    def forward(self, x, alpha=None):
        #   传入x尺寸为B*D*H*W*C
        x_temp = x.clone()
        x = self.norm(x_temp)
        B, D, H, W, C = x.shape
        x_re = rearrange(x, 'B D H W C -> C (B D) (H W)')
        soft_assign = self.euclid_dis(x_re.contiguous(), self.cluster_center)  # 这里返回的是向量间的距离
        # 若是输入b*w*h*c的矩阵，返回则是b*w*h*1
        x_distance = rearrange(soft_assign, 'C (B D) CN -> B D C CN', D=D)
        x_distance_assign = self.assign_func(x_distance, alpha)
        x_rec = []
        # kk = self.cluster_center.clone()
        # # x_rec = torch.empty((B, D, H*W)).cuda()
        #
        # x_distance_assign_temp = rearrange(x_distance_assign, 'B D C CN -> C B D CN')
        # for i, j in zip(x_distance_assign_temp, kk):
        #     temp = i @ j
        #     x_rec.append(temp)
        #
        # x_rec = torch.stack(x_rec)
        # x_rec = rearrange(x_rec, 'C B D (H W) -> B D H W C', H=H, W=W)
        cluster_dist = self.self_similarity()
        return x_distance, x_distance_assign, cluster_dist, x_rec

# if __name__ == "__main__":
#     x = torch.zeros((1, 2, 768, 25))
#     x = x.permute(0, 2, 3, 1).contiguous()
#     x_re = x.view(x.shape[0], -1, x.shape[1])
#     soft_assign = PosSoftAssign(0,8)
#     xx  =  torch.rand(10)
#     soft_xx = soft_assign(xx)
#     print(xx)
#     print(soft_xx)
