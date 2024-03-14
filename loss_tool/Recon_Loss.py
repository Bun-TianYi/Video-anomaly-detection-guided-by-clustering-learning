from einops import rearrange
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


class Recon_Loss(nn.Module):
    """
    应当传进两个视频，一个为原始视频，一个为重建视频
    视频格式应当为‘B D W H C’
    采用L1范数（防止出现网络退化问题）
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


    def forward(self, x, target):
        _, _, D, H, W = target.size()
        if D % self.patch_size[0] != 0:
            target = F.pad(target, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        assert x.shape == target.shape
        x = rearrange(x, 'B C D W H -> B D W H C')
        target = rearrange(target, 'B C D W H -> B D W H C')
        loss = F.l1_loss(x, target)

        return loss
