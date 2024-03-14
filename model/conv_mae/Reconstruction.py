import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 2, n_channel =3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                # 此为卷积操作，值得注意padding参数，该参数决定对特征矩阵的补零操作。
                # in_channels为输入的通道数（RGB为3通道），反之，out_channels即为输出通道数
                # 顺便，这个Conv2D是一个方法，单独调用这个方法设定参数是在实例化这个方法
                torch.nn.BatchNorm2d(intOutput),
                # 这个方法是对传入图像进行均值方差标准化
                torch.nn.ReLU(inplace=False),
                # 非必要不要进行inplace操作，这会将ReLu的输出覆盖到输入中
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
            # 这是设定了两层卷积层方法
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        # 这里在设定模型的U-net结构的decode部分
        # 也就是每层卷积层卷两次，最后一层卷积层卷完后不进行Relu操作
        self.moduleConv1 = Basic(n_channel*t_length, 64)
        # 设定模型卷积层（对之前的卷积层方法进行实例化）
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 设定模型pooling操作
        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        #self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):
        # 传入一张图
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        # 到这里完成encode的过程
        return tensorConv4
        # 返回一个编码完成后的图像
    
    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 2, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            # 最后一层生成
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            # 上采样模型
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                # 反卷积方法层，该层可以理解为编码层中的max_pooling层
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 512)  # 先实例化卷积方法
        self.moduleUpsample4 = Upsample(512, 512)   # 实例化上采样方法

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 256)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 128)

        self.moduleDeconv1 = Gen(128,n_channel*t_length,64)
        
        
        
    def forward(self, x):
        # 这个函数是对解码进行模块组装
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        
        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        
        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        
        output = self.moduleDeconv1(tensorUpsample2)

                
        return output
    


class convAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 2, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        # 这里也只是在实例化方法而已，还没有开始训练
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
        # 实例化方法内的各个参数

    def forward(self, x, keys,train=True):

        fea = self.encoder(x)
        # 居然先进行编码操作
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        
        #test
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss
        
                                          



    