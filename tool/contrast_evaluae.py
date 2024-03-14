# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import pandas as pd
from einops import rearrange
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import dataset.utils_dataset as dt
from misc import utils
from model import swin_transformer as encoder
from utils import distritributed_model as dp
from model import swin_decoder as decoder
from model import backbone
from loss_tool.Recon_Loss import Recon_Loss
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Swin Transformer', add_help=False)

    # Model parameters

    parser.add_argument('--frame_num', default=4, type=int, help="请输入一次读取的视频帧数")
    parser.add_argument('--patch_size', default=(2, 4, 4), type=tuple, help="请输入patch编码的大小")
    # Training/Optimization parameters

    parser.add_argument('--weight_decay', type=float, default=0.024, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.02, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=2, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00008, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Misc
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    #======================上海科技大学数据集参数====================#
    parser.add_argument('--data_path', default='F:/ShanghaiTech/testing/test_frames', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--label_path', default='F:\ShanghaiTech/testing/test_frame_mask', type=str)
# # ========================Avenue数据集参数========================#
#     parser.add_argument('--data_path', default='F:/Avenue_Dataset/test', type=str,
#                         help='Please specify path to the ImageNet training data.')
#     parser.add_argument('--label_path', default='F:/Avenue_Dataset/testing_label', type=str)

    #=======================Ped2数据集参数======================#
    # parser.add_argument('--data_path', default='F:\Ped2/test/frames', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:\Ped2/test_label', type=str)


    parser.add_argument('--model_pretrain', default='F:\My-repository\My_Model\log_dir\checkpoint0.pth',
                        type=str, help='训练好的模型路径')
    parser.add_argument('--output_dir', default="eval_output", type=str, help='保存输出文件路径')
    parser.add_argument('--saveckp_freq', default=15, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--ispredict", default=False, type=bool, help="更改网络模式")
    return parser




def predict_main(args):
    dp.init_distributed_mode(args)
    dp.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(dp.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ========准备数据集========#
    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.data_path, transform=transform,
                            frames_num=args.frame_num, label_folder=args.label_path,
                            istest=True)
    # kk = dataset.__getitem__(0)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(  # 将分布式采样的数据使用dataloader进行转换
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"共有{len(dataset)}个视频加载完成 ")

    # ============构建网络==============#
    model = backbone.Mymodel(args, ispredict=args.ispredict, iscluster=True)
    model.cuda()

    # =============预训练模块================================#
    utils.load_pretrain_model(args.model_pretrain, model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 将网络部署至分布式环境
    print("网络初始化完成")

    # =============准备loss函数（暂且只有重建loss）============#
    recon_loss = nn.MSELoss(reduction='none')
    loss_log = []
    start_time = time.time()
    print("开始训练")
    data_iter = 0
    for epoch in range(1):
        model.eval()
        model.module.cluster_loss_on()
        model.module.encoder_compatness()
        data_loader.sampler.set_epoch(epoch)
        data_iter = predict(model, recon_loss, data_loader, dataset, data_iter)


def predict(model, recon_loss, data_loader, dataset, data_iter):

    scene_dict = {}
    scene_label = {}
    auc = 0
    for it, (images, idx, label, scene_num) in enumerate(data_loader):
        # move images to gpu

        predict_label = []
        truth_label = []
        index = 0
        while label.dim() != 2:
            label = torch.squeeze(label, 0)

        print(str(it) + '/' + str(len(dataset)))  # 打印进度信息
        while index + args.frame_num< len(images[0, 0, :]):
            clip = images[:, :, index: index + args.frame_num]
            if args.ispredict:
                label_clip = label[:, index + args.frame_num]
            else:
                label_clip = label[0, index: index + args.frame_num]
            for it in range(args.batch_size -1):
                if index + args.frame_num + 1< len(images[0, 0, :]):
                    index = index + args.frame_num
                    temp_clip = images[:, :, index: index + args.frame_num]
                    clip = torch.cat((clip, temp_clip), dim=0)
                    # print(index + args.frame_num)
                    if args.ispredict:
                        label_clip = torch.cat((label_clip, label[:, index + args.frame_num]), dim=0)
                    else:
                        label_clip = torch.cat((label_clip, label[0, index:index + args.frame_num]), dim=0)
                else:
                    break
            index = index + args.frame_num
            if args.ispredict:
                true_video = clip[:, :, -1:].cuda()
                clip = clip[:, :, 0: 4].cuda()
            else:
                true_video = clip[:, :, 0:].cuda()
                clip = clip.cuda()

            #images = rearrange(images, 'B C D H W -> B D C H W')
            recon, cluster_loss, space_loss, cluster_assgin, sapce_assgin, _, _ = model(clip)
            # utils.save_tensor_video(recon, output_dir='video_show_test')
            #utils.save_tensor_video(recon-clip, output_dir='test_video_show')
            # utils.save_tensor_video(recon)

            # utils.tensor_medfit(recon[:, :, -1:])
            # utils.save_tensor_video(recon[:, :, 0:] - clip[:, :, 0:], output_dir="video_show_origin", save_name=str('%03d' % (index + args.frame_num))+'.jpg')
            # cluster_loss_ = torch.mean(cluster_loss).detach().cpu()
            # space_loss_ = torch.mean(space_loss).detach().cpu()

            # true_video = clip[:, :, 0]

            #true_video = clipeze()

            #cluster_loss = torch.me
            #true_video = true_video.squean(cluster_loss, [2, 3, 4])
            # temp = recon[:, :, 0]
            if args.ispredict:
                loss = recon_loss(recon, true_video)
            else:
                loss = recon_loss(recon[:, :, 0:], true_video)
            # loss = utils.tensor_medfit(recon[:, :, 0] - true_video)
            loss = rearrange(loss, 'B C D H W -> B D C H W')
            loss_frame = torch.mean(loss, dim=4).mean(dim=3).mean(dim=2)
            loss_frame = loss_frame.tolist()
            loss_frame = sum(loss_frame,[])
            kk = utils.psnr(loss_frame)
            # kk2 =cluster_loss_  + space_loss_
            psnr_frame = kk
            # for id in range(len(images[0, :])):
            #
            #     if id + args.frame_num < len(images[0, :]):
            #         img = images[:, id: id + args.frame_num]
            #     else:
            #         break
            #     img = rearrange(img, 'B D C H W -> B C D H W')
            #     id = id + args.frame_num
            #     recon = model(img)
            #     loss = recon_loss(recon, img)
            #     loss_frame.append(torch.mean(loss).item())


            # plt.plot(psnr_frame, 'r', label='Training loss')
            # plt.title('Training loss')
            # plt.show()
            predict_label.extend(psnr_frame)
            label_clip = label_clip.tolist()
            truth_label.extend(label_clip)
            kk = 0



        assert len(predict_label) == len(truth_label)
        predict_label = np.array([utils.anomly_score(predict_label)])[0]

        truth_label = np.array([truth_label])[0]
        scene_num = scene_num[0]
        if scene_num in scene_dict:
            scene_dict[scene_num] = np.append(scene_dict[scene_num], predict_label)
            scene_label[scene_num] = np.append(scene_label[scene_num], truth_label)
        else:
            scene_dict.update({scene_num: predict_label})
            scene_label.update({scene_num: truth_label})

    for idx, key in enumerate(scene_dict):
        # kk = utils.anomly_score(scene_dict[key])
        temp = roc_auc_score(scene_label[key], scene_dict[key])
        # temp = roc_auc_score(scene_label[key], kk)
        # label_record = scene_label[key]
        print(str(key) + "场景下的auc值为:" + str(temp))
        plt.title('Comparison of two anomaly detection paradigms')
        df = pd.read_csv('temp.csv', index_col=0)
        data1 = df.values.tolist()

        plt.plot(scene_dict[key][0:24])
        plt.plot(data1)

        plt.ylabel('Abnormal score')
        plt.xlabel('frames')
        plt.show()


        auc = auc + temp

    # auc.append(roc_auc_score(truth_label, predict_label))
    # print(roc_auc_score(truth_label, predict_label))
    auc = auc/(idx + 1)
    print('AUC值为' + str(auc))
    return data_iter


if __name__ == '__main__':
    # 这里需要对数据加强模块进行一次大改动，需要将逐像素翻转变成对整个视频段的处理
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    predict_main(args)
