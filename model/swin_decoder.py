import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from model.I3D import *

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Patchdebed3D(nn.Module):
    """
    特征解码操作，将特征反向debed
    """
    def __init__(self, patch_size=(2,4,4), in_chans=96, embed_dim=3, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()        # 这里注意一下，输入的必须是 (B C D H W)，这里D表示时间，也不知道为什么，反正人这么写了
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        # x = rearrange(x, 'N C D H W ->N D C H W')
        # x = self.norm(x)
        # x = rearrange(x, 'N D C H W ->N C D H W')
        # if self.norm is not None:
        #     D, Wh, Ww = x.size(2), x.size(3), x.size(4)
        #     x = x.flatten(2).transpose(1, 2)
        #     x = self.norm(x)
        #     x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x



class up_sampling(nn.Module):
    """
    将特征进行上采样，采样方法为原论文的下采样过程反过来
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        #self.dim = dim
        #self.unreduction = nn.Linear(dim, 2 * dim, bias=False)
        #self.norm = norm_layer(dim)
        self.proj = nn.ConvTranspose3d(dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        #self.bn = nn.BatchNorm3d(10, affine=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        #B, D, H, W, C = x.shape
        x = rearrange(x, 'B D H W C -> B C D H W')
        x = self.proj(x)
        x = rearrange(x, 'B C D H W -> B D H W C')

        return x


class SwinDecoder(nn.Module):
    def __init__(self, in_chans, patch_size):
        super().__init__()
        self.in_chans = in_chans
        self.upsampling = nn.ModuleList()
        self.conv_bn_relu = nn.ModuleList()
        self.time_down_sample = nn.ModuleList()
        up_chans = int(in_chans/2)
        for i in range(3):
            temp = up_sampling(up_chans)
            up_chans = int(up_chans/2)
            self.upsampling.append(temp)

        down_chans = in_chans
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv3d(in_channels=down_chans*2, out_channels=down_chans, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(down_chans, eps=0.001, momentum=0.01),
                nn.ReLU(),
                nn.Conv3d(in_channels=down_chans, out_channels=int(down_chans/2), kernel_size=(1, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(int(down_chans/2), eps=0.001, momentum=0.01),
                nn.ReLU()
            )
            down_chans = int(down_chans/2)
            self.conv_bn_relu.append(layer)
        self.conv_bn_relu.append(
            nn.Sequential(
                nn.Conv3d(in_channels=down_chans * 2, out_channels=down_chans, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(down_chans, eps=0.001, momentum=0.01),
                nn.ReLU(),
                nn.Conv3d(in_channels=down_chans, out_channels=down_chans, kernel_size=(1, 3, 3),
                          stride=(1, 1, 1),
                          padding=(0, 1, 1)),
                nn.BatchNorm3d(int(down_chans), eps=0.001, momentum=0.01),
                nn.ReLU()
            )
        )
        self.patchdebed = Patchdebed3D(patch_size=patch_size)

    def forward(self, x, x_drec):
        x_drec.reverse()
        x_drec = x_drec[1:]

        for idx, (up, conv) in enumerate(zip(self.upsampling, self.conv_bn_relu)):
            if idx == 0:
                x = torch.cat((x, x_drec[idx]), dim=4)
                x = rearrange(x, 'B D H W C -> B C D H W')
                x = conv(x)
                x = rearrange(x, 'B C D H W -> B D H W C')
                x = up(x)

        x = rearrange(x, 'B D H W C -> B C D H W')
        x = self.conv_bn_relu[-1](x)
        x = self.patchdebed(x)

        return x