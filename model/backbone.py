import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

from mmcv.runner import load_checkpoint
from mmaction.utils import get_root_logger
# from ..builder import BACKBONES
from model import unet3D as unet
from model import swin_transformer as encoder
from model import swin_decoder as decoder
from model import swin_decoder_predict as decoder_predict
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from .conv_mae.Reconstruction import *
from model.cluster import EuclidDistance_Assign_Module as cluster
from model.cluster import Space_EuclidDistance_Assign_Module as space_cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from misc import utils




class Mymodel(nn.Module):
    def __init__(self, args, iscluster=False, ispredict=False):
        super().__init__()
        #self.backbone = convAE(3, t_length=args.frame_num, memory_size=10, feature_dim=512, key_dim=512)
        # self.backbone = unet.UNet3D()
        self.iscluster = iscluster
        self.isCompactness = False
        self.encoder = encoder.SwinTransformer3D(patch_size=args.patch_size)
        if ispredict:
            self.decoder = decoder_predict.SwinDecoder(in_chans=192, patch_size=args.patch_size)
        else:
            self.decoder = decoder_predict.SwinDecoder(in_chans=192, patch_size=args.patch_size, ispredict=ispredict)
        self.cluster1 = cluster(feature_dim=192, cluster_num=1024,  soft_assign_alpha=16.0)
        self.space_cluster = space_cluster(feature_dim=192, cluster_num=128, soft_assign_alpha=32.0, space_size=28
                                           )
        # self.cluster2 = cluster(feature_dim=192, cluster_num=1024, soft_assign_alpha=16.0)
        #self.cluster3 = cluster(feature_dim=384, cluster_num=240,  soft_assign_alpha=32.0)
        self.norm = nn.LayerNorm(192)
        for name, parameter in self.named_parameters():
            if 'cluster' in name:
                parameter.requires_grad = False
                print(name + '梯度已停止更新')

    def cluster_loss_on(self):
        self.iscluster = True

    def cluster_on(self):
        for name, parameter in self.named_parameters():
            if 'cluster' in name:
                if 'identity_matrix' not in name:
                    parameter.requires_grad = True
                    print(name + '梯度已开启更新')
        self.iscluster = True


    def cluster_center_on(self):
        for name, parameter in self.named_parameters():
            if 'cluster_center' in name:
                parameter.requires_grad = True

    def cluster_center_off(self):
        for name, parameter in self.named_parameters():
            if 'cluster_center' in name:
                parameter.requires_grad = False
                print(name + '梯度更新完毕')


    def encoder_compatness(self):
        self.isCompactness = True


    def forward(self, x):
        x, x_dec, x_drec = self.encoder(x)
        #x = rearrange(x, 'B C D W H -> B ( C D ) W H')
        x = rearrange(x, 'B C D H W -> B D H W C')
        x_assign1 = 0
        xf_assign = 0
        if self.iscluster:
            x_temp = x.detach()
            if self.isCompactness:
                x_temp = x
                x_distance1, x_assign1, self_dist1, x , feature, feature_label = self.cluster1(x_temp)
                xf_distance, xf_assign, space_self_dist, x0 = self.space_cluster(x_temp)
            else:
                x_distance1, x_assign1, self_dist1, x0 = self.cluster1(x_temp)
                xf_distance, xf_assign, space_self_dist, x0 = self.space_cluster(x_temp)
            space_cluster_loss = torch.norm(xf_distance * xf_assign)
            # space_cluster_loss = 0
            # self_dist = torch.mean(self_dist)*0.01
            # space_self_dist = torch.mean(space_self_dist)*0.01
            cluster_loss = torch.norm(x_distance1 * x_assign1) #+ torch.mean(x_distance3 * x_assign3)
            # space_cluster_loss = 0


        else:
            cluster_loss = None
            space_cluster_loss = None
            B, D, H, W, C = x.shape
            x_re = rearrange(x, 'B D H W C -> (B D H W) C')
            feature = x_re
            feature_label = torch.zeros((B*D*H*W))

        #========可视化，不用请注释=============#
        # tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate=200)
        # feature = feature.detach().clone().cpu().numpy()
        # feature_label = feature_label.detach().clone().cpu().numpy()
        # result = tsne.fit_transform(feature)
        # fig = utils.plot_embedding(result, feature_label, title='tsne')
        # # plt.show()
        # sys.exit()

        # x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        x = self.decoder(x, x_dec, x_drec)
        # x, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.backbone(
        #     x, m_items, True)
        # x = rearrange(x, 'B ( C D ) W H -> B C D W H', C=3)
        #x = self.backbone(x)

        return x, cluster_loss, space_cluster_loss, 0, 0, feature, feature_label
