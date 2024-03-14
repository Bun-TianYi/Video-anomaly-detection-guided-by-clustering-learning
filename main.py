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
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

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

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Swin Transformer', add_help=False)

    # Model parameters

    parser.add_argument('--frame_num', default=10, type=int, help="请输入一次读取的视频帧数")
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
    parser.add_argument('--batch_size_per_gpu', default=3, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=2, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.000008, type=float, help="""Learning rate at the end of
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
    parser.add_argument('--data_path', default='F:/My-repository/My_Model/frames', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--model_pretrain', default='F:\My-repositoryMy_Model\pretrain_model\checkpoint9.pth',
                        type=str, help='预训练模型存放处')
    parser.add_argument('--output_dir', default="log_dir", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    dp.init_distributed_mode(args)
    dp.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(dp.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ========准备数据集========#
    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.data_path, transform=transform, frames_num=args.frame_num)
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

    # model_encoder = encoder.SwinTransformer3D(patch_size=args.patch_size)
    # model_decoder = decoder.SwinDecoder(in_chans=768, patch_size=args.patch_size)  # 这里未对代码进行比较好的包装，后续考虑进行重构
    #m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).cuda()
    #model = backbone.Mymodel(args)
    model = backbone.Mymodel(args)
    model.cuda()

    # =============预训练模块================================#
    #utils.load_pretrain_model(args.model_pretrain, model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 将网络部署至分布式环境
    print("网络初始化完成")

    # =============准备loss函数（暂且只有重建loss）============#
    recon_loss = nn.MSELoss(reduction='none')
    loss_log = []

    # =============构建优化器==========#
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.min_lr,
                                                          last_epoch=-1)
    plt.ion()
    # ===============初始化日志文件路径================#
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_file_path = os.path.join(args.output_dir, 'exp.log')
    logger = utils.get_logger(logging_file_path)
    # =============开始训练============#
    start_time = time.time()
    print("开始训练")
    data_iter = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch % 1 == 0:  # 每个epoch保存一次
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint'+str(epoch)+'.pth'))
        data_loader.sampler.set_epoch(epoch)

        # ==============这里开始正式训练模型===============#
        train_stats, data_iter= train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter, dataset,
                                      logger, args, loss_log)


# =============训练一个epoch=================#
def train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter, dataset, logger, args, loss_log):
    """
    该函数中，最终输出视频均为‘B C D H W’的形式
    """

    loop = tqdm(data_loader, leave=False)
    for it, (video, idx) in enumerate(loop):
        video = video.cuda()
        #video = rearrange(video, 'B C D W H -> B ( D C ) W H')
        if data_iter == 500:
            model.module.cluster_on()

        recon_video, cluster_loss, space_loss = model(video)
        #recon_video2 = rearrange(recon_video, 'B ( D C ) W H -> B C D W H', C=3)
        if data_iter % 10 == 0:
            utils.save_tensor_video(video, output_dir='video_show_origin')
            utils.save_tensor_video(recon_video)

        optimizer.zero_grad()
        loss_pixel = torch.mean(recon_loss(recon_video, video))
        if cluster_loss is not None:
            cluster_loss = torch.mean(cluster_loss)
            space_loss = torch.mean(space_loss)
            loss = loss_pixel + cluster_loss + space_loss
        else:
            loss = loss_pixel
        loss.backward(retain_graph=True)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)  # 若出现loss消失现象则停止训练
            sys.exit(1)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info('Epoch:[{}/{}]\t batch:[{}/{}]\t loss={:.5f}\t lr={:.3f}'.format(epoch, args.epochs, it, len(loop), loss, lr))
        if data_iter > 2500:
            loss_log.append(float(loss.detach().cpu().numpy()))
            plt.clf()
            plt.plot(loss_log, 'r', label='Training loss')
            plt.title('Training loss')
            plt.legend()
            #plt.show()
            plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.ioff()  # 关闭画图窗口


        data_iter = data_iter + 1
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item())

        # for name, parms in model.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     if parms.grad is not None:
        #         print('-->grad_value', parms.grad.cpu().numpy())
        #         hh = parms.grad.cpu().numpy()
        #     print("===")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()
    scheduler.step()
    return 0, data_iter


def hookfunc(model, grad_input, grad_output):
    print('model:', model)
    print('grad input:', grad_input)
    print('grad output:', grad_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
