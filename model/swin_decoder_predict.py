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


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x



# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # self.downsample = nn.Linear(dim, int(dim/2), bias=False)
        self.downsample = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        x_copy = x
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
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

        self.proj = nn.Sequential(
            nn.ConvTranspose3d(96, 192, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)),
            # nn.BatchNorm3d(48),
            nn.GELU(),
            nn.Conv3d(192, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GELU(),
            nn.ConvTranspose3d(96, 3, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)),
            # nn.Sigmoid()
        )
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
        # self.dim = dim
        # self.unreduction = nn.Linear(dim, 2 * dim, bias=False)
        # self.norm = nn.LayerNorm(int(dim*2))
        self.proj = nn.Sequential(nn.ConvTranspose3d(dim, int(dim/2), kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                                  # nn.BatchNorm3d(int(dim/2)),
                                  nn.GELU()
                                  )
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
        # B, D, H, W, C = x.shape

        # x = self.unreduction(x)
        # x = self.norm(x)
        # # padding
        # x0 = x[:, :, :, :, 0:int(C/2)].clone()
        # x1 = x[:, :, :, :, int(C/2): C].clone()
        # x2 = x[:, :, :, :, C: int(3*C/2)].clone()
        # x3 = x[:, :, :, :, int(3*C/2): 2*C].clone()
        # x_temp = torch.zeros((B, D, H*2, W*2, int(C/2))).cuda()
        # x_temp[:, :, 0::2, 0::2, :] = x0  # B D H/2 W/2 C
        # x_temp[:, :, 1::2, 0::2, :] = x1  # B D H/2 W/2 C
        # x_temp[:, :, 0::2, 1::2, :] = x2  # B D H/2 W/2 C
        # x_temp[:, :, 1::2, 1::2, :] = x3  # B D H/2 W/2 C
        # x = x_temp
        #torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C




        return x


class SwinDecoder(nn.Module):
    def __init__(self, in_chans, patch_size, pretrained=None,
                 pretrained2d=True,
                 depths=[6, 3],
                 num_heads=[12, 6],
                 window_size=(8, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 norm_layer=nn.LayerNorm,
                 ispredict=True,
                 use_checkpoint=False):
        super().__init__()
        self.in_chans = in_chans
        self.upsampling = nn.ModuleList()
        self.ST_layers = nn.ModuleList()
        self.I3D_layers = nn.ModuleList()
        up_chans = in_chans
        # self.upsampling.append(nn.Identity())

        for i in range(1):
            temp = up_sampling(up_chans)
            up_chans = int(up_chans/2)
            self.upsampling.append(temp)
        self.upsampling.append(nn.Identity())

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dim = in_chans
        # self.ST_layers.append( BasicLayer(
        #         dim=192,
        #         depth=depths[0],
        #         num_heads=num_heads[0],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[0:sum(depths[:1])],
        #         norm_layer=norm_layer,
        #         downsample=None,
        #         use_checkpoint=use_checkpoint))
        for i_layer in range(2):
            layer = BasicLayer(
                dim=int(dim / 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.ST_layers.append(layer)

        out_channel = [
            # [256, 112, 256, 32, 128, 128], # 输出通道为768
            # [256, 112, 256, 32, 128, 128],  # 输出通道为768
            # [128, 96, 128, 32, 64, 64],  # 输出通道为384
            [32, 64, 96, 16, 32, 32],  # 输出总通道为192
            [16, 32, 48, 9, 16, 16],  # 输出通道为96
        ]
        up_chans = in_chans
        # self.I3D_layers.append(nn.Sequential(
        #         InceptionModule(
        #             in_channels=192,
        #             out_channels=out_channel[0],
        #             name='inception_i3d' + str(0)
        #         ),
        #     ))
        for i_layer in range(2):
            layer = nn.Sequential(
                InceptionModule(
                    in_channels=int(up_chans / 2 ** i_layer),
                    out_channels=out_channel[i_layer],
                    name='inception_i3d' + str(i_layer)
                ),
            )
            self.I3D_layers.append(layer)


        # self.jump = nn.ModuleList()
        # self.jump.append(nn.Sequential(
        #     nn.Conv3d(in_channels=96, out_channels=96,
        #               kernel_size=(1, 1, 1), stride=1),
        #     nn.BatchNorm3d(96),
        #     nn.GELU()
        # ))
        # jump_chans = in_chans
        # self.jump = nn.ModuleList()
        # self.jump.append(nn.Sequential(
        #         nn.Conv3d(in_channels=768, out_channels=768,
        #                   kernel_size=(1, 1, 1), stride=1),
        #         nn.BatchNorm3d(768),
        #         nn.GELU()
        #     ))
        # for i_layer in range(4):
        #     layer = nn.Sequential(
        #         nn.Conv3d(in_channels=int(jump_chans / 2 ** i_layer), out_channels=int(jump_chans / 2 ** i_layer),
        #                   kernel_size=(1, 1, 1), stride=1),
        #         nn.BatchNorm3d(int(jump_chans / 2 ** i_layer)),
        #         nn.GELU()
        #     )
        #     self.jump.append(layer)
        # self.timedebd = nn.ConvTranspose3d(in_channels=192, out_channels=192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        if ispredict:
            self.timedebd = nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        else:
            self.timedebd = nn.ConvTranspose3d(in_channels=192, out_channels=192, kernel_size=(2, 1, 1),
                                               stride=(2, 1, 1))
        self.norm = nn.LayerNorm(96)
        self.patchdebed = Patchdebed3D(patch_size=patch_size)

    def forward(self, x, x_dec, x_drec):
        x = rearrange(x, 'B D H W C -> B C D H W')
        x = self.timedebd(x)
        x = rearrange(x, 'B C D H W -> B D H W C')
        # x_drec.reverse()
        # x_drec = x_drec[1:]

        for idx, (up, attn, conv) in enumerate(zip(self.upsampling, self.ST_layers, self.I3D_layers)):
            # x_drec_temp = x_drec[idx].clone()
            # if idx == 3:
            #     x_drec_temp[:] = 0
            # x_drec_temp = rearrange(x_drec_temp, 'B D H W C -> B C D H W')
            # x_drec_temp = jump(x_drec_temp)
            # x_drec_temp = rearrange(x_drec_temp, 'B C D H W -> B D H W C')
            # x = torch.cat((x, x), dim=4)
            # x = x + x_drec_temp
            x = rearrange(x, 'B D H W C -> B C D H W')
            x_conv = conv(x)
            x_attn = attn(x)
            x_conv = x_conv * x_attn
            x = x_attn + x_conv + x
            x = rearrange(x, 'B C D H W -> B D H W C')
            # x = x + x_drec_temp
            x = up(x)

        # x_rec_temp = x_rec.clone()
        # x_rec_temp[:, -1] = 0
        x = self.norm(x)
        x = rearrange(x, 'B D H W C -> B C D H W')

        # x = self.jump[-1](x)
        # x = x + x_dec
        x = self.patchdebed(x)

        return x