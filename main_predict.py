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
import pandas as pd
from sklearn.metrics import roc_auc_score
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
import pandas
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
import timm
import timm.optim
import timm.scheduler

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
    parser.add_argument('--batch_size_per_gpu', default=4, type=int,
                        help='Per-GPU batch-size : number of distinct i mages loaded on one GPU.')
    parser.add_argument('--epochs', default=120, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=2, type=int, help="""Num01 ber of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.000006, type=float, help="""Learning rate at the end of
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

    # ==============该段为ShanhaiTech数据集的参数================#
    parser.add_argument('--data_path', default='F:/My-repository/My_Model/frames', type=str,
                            help='Please specify path to the ImageNet training data.')

    # parser.add_argument('--test_data_path', default='F:/ShanghaiTech/testing/frames', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:\ShanghaiTech/testing/test_frame_mask', type=str)
    # ==========================================================#
    # ==============该段为Avenue数据集的参数=================#
    # parser.add_argument('--data_path', default='F:/Avenue_Dataset/training', type=str,
    #                     help='Please specify path to the ImageNet training data.')

    #==============该段为Ped2数据集=====================#

    # parser.add_argument('--data_path', default='F:\Ped2/training/frames', type=str,
    #                     help='Please specify path to the ImageNet training data.')

    # parser.add_argument('--test_data_path', default='F:/Avenue_Dataset/testing', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:/Avenue_Dataset/testing_label', type=str)
    # ==========================================================#


    parser.add_argument('--test_batch_size_per_gpu', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

    parser.add_argument('--model_pretrain',
                        default='F:\My-repository\My_Model\log_dir\checkpoint63.pth',
                        type=str, help='预训练模型存放处')
    parser.add_argument('--output_dir', default="log_dir", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--start", default=0, type=int, help="训练断点")
    parser.add_argument("--ispredict", default=False, type=bool, help="训练断点")

    return parser


def train_dino(args):
    dp.init_distributed_mode(args)
    dp.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(dp.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ========准备数据集========#
    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.data_path, transform=transform, frames_num=args.frame_num, index_num=3, image_format='jpg')
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
    # m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1).cuda()

    # model = backbone.Mymodel(args)
    model = backbone.Mymodel(args, ispredict=args.ispredict, iscluster=False)
    # utils.load_pretrain_model(args.model_pretrain, model)
    model.cuda()

    # =============预训练模块================================#
    utils.load_pretrain_model(args.model_pretrain, model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # 将网络部署至分布式环境

    print("网络初始化完成")

    # =============准备loss函数（暂且只有重建loss）============#
    recon_loss = nn.MSELoss(reduction='none')
    loss_log = []

    # =============构建优化器==========#
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.02)  # to use with ViTs

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.min_lr,
    #                                                       last_epoch=-1)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer, t_initial=args.epochs, lr_min=args.min_lr,
                                                 warmup_t=0, warmup_lr_init=0.000001)
    plt.ion()
    # ===============初始化日志文件路径================#
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_file_path = os.path.join(args.output_dir, 'exp.log')
    logger = utils.get_logger(logging_file_path)
    # =============开始训练============#
    start_time = time.time()
    auc_recode = 0
    print("开始训练")
    data_iter = 0
    auc_list = []
    # df = pd.DataFrame(auc_list)
    # df.to_excel("auc_record.xlsx", index=False)
    model.train()
    for epoch in range(args.start, args.epochs):

        if epoch % 1 == 0:  # 每个epoch保存一次
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint' + str(epoch) + '.pth'))
        data_loader.sampler.set_epoch(epoch)

            # ==============这里开始正式训练模型===============#
        train_stats, data_iter = train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter,
                                                 dataset,
                                                 logger, args, loss_log)
        auc = 0
    #     if epoch % 4 == 0:
    #         auc = test(args, model, auc_recode)
    #         auc_list.append(auc)
    #     if auc > auc_recode:
    #         auc_recode = auc
    # df = pd.DataFrame(auc_list)
    # df.to_excel("auc_record.xlsx", index=False)




# =============训练一个epoch=================#
def train_one_epoch(model, recon_loss, data_loader, optimizer, scheduler, epoch, data_iter, dataset, logger, args,
                    loss_log):
    """
    该函数中，最终输出视频均为‘B C D H W’的形式
    """

    loss_record = 0
    issave = True
    loop = tqdm(data_loader, leave=False)
    for it, (video, idx) in enumerate(loop):
        if args.ispredict:
            predict_frame = video[:, :, -1:, :, :].detach().clone().cuda()
            video = video[:, :, 0:4, :, :]
            video = video.cuda()
        else:
            predict_frame = video.detach().clone().cuda()
            video = video
            video = video.cuda()

        # video = rearrange(video, 'B C D W H -> B ( D C ) W H')
        if data_iter < 0:
            optimizer.param_groups[0]['lr'] = 0.0004
        elif data_iter == 0:
            optimizer.param_groups[0]['lr'] = args.lr

        if data_iter == 0:
            model.module.cluster_on()
            model.module.cluster_center_on()

        if data_iter == 0:
            model.module.encoder_compatness()
            # print("编码器开始紧致化")
        if data_iter == 0:
            model.apply(utils.freeze_bn)
        # if data_iter >=5000 and data_iter % 10 ==0:
        #     model.module.cluster_center_on()

        recon_video, cluster_loss, space_loss, self_dist, sapce_self_dist, feature, feature_label = model(video)


        # recon_video2 = rearrange(recon_video, 'B ( D C ) W H -> B C D W H', C=3)
        if data_iter % 10 == 0:
            utils.save_tensor_video(predict_frame, output_dir='video_show_origin')
            utils.save_tensor_video(recon_video)

        predict_frame = predict_frame
        # recon_video = recon_video.squeeze()
        optimizer.zero_grad()
        if args.ispredict:
            loss_pixel = torch.norm(recon_loss(recon_video, predict_frame))
        else:
            loss_pixel = torch.norm(recon_loss(recon_video, predict_frame))

        if cluster_loss is not None:
            cluster_loss = torch.mean(cluster_loss)
            # cluster_loss = torch.max(cluster_loss - self_dist + 0.05, torch.tensor(0))
            # space_loss = torch.mean(space_loss)
            # space_loss = 0
            # space_loss = torch.max(space_loss - sapce_self_dist + 0.05, torch.tensor(0))

            loss = loss_pixel + cluster_loss + space_loss
        else:
            loss = loss_pixel
        # for idx, (name, param) in enumerate(model.named_parameters()):
        #     print(str(idx) + name)
        torch.autograd.set_detect_anomaly(True)
        if abs(loss_record - loss) > 10 and issave:
            utils.save_tensor_video(predict_frame, output_dir='bug_data_detect')
            issave = False
        else:
            loss_record = loss.item()

        loss.backward()

        # if data_iter >=5000 and data_iter % 10 == 0:
        #     model.module.cluster_center_off()

        # if data_iter >= 3500 and data_iter % 10 == 0:
        #     model.module.cluster_center_off()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)  # 若出现loss消失现象则停止训练
            sys.exit(1)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(
            'Epoch:[{}/{}]\t batch:[{}/{}]\t loss={:.5f}\t lr={:.7f}'.format(epoch, args.epochs, it, len(loop), loss,
                                                                             lr))
        loss_log.append(float(loss_pixel.detach().cpu().numpy()))
        if data_iter > 2500:
            # loss_log.append(float(loss.detach().cpu().numpy()))
            plt.clf()
            plt.plot(loss_log, 'r', label='Training loss')
            plt.title('Training loss')
            plt.legend()
            # plt.show()
            plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.ioff()  # 关闭画图窗口

        data_iter = data_iter + 1
        loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item(), lr=lr, recon_loss=loss_pixel.item())
        # loop.set_postfix(loss=loss.item(), lr=lr, recon_loss=loss_pixel.item()
        #                  , cluster_loss=cluster_loss.item(), space_loss=space_loss.item())
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
        if data_iter % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint' + str(epoch) + '.pth'))
        # if data_iter ==60:
        #     save_path = 'loss_record'
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)
        #     loss_log = np.array(loss_log)
        #     np.save(os.path.join(save_path, 'mix_Shan' + '.npy'), loss_log)
        #     sys.exit()
    scheduler.step(epoch)
    return 0, data_iter


def hookfunc(model, grad_input, grad_output):
    print('model:', model)
    print('grad input:', grad_input)
    print('grad output:', grad_output)


def main_predict(args, model, auc_recode):
    # ========准备数据集========#
    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.test_data_path, transform=transform,
                            frames_num=args.frame_num, label_folder=args.label_path,
                            istest=True, index_num=3, image_format='jpg')
    # kk = dataset.__getitem__(0)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(  # 将分布式采样的数据使用dataloader进行转换
        dataset,
        sampler=sampler,
        batch_size=args.test_batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"共有{len(dataset)}个视频加载完成 ")

    # =============准备loss函数（暂且只有重建loss）============#
    recon_loss = nn.MSELoss(reduction='none')
    print("开始测试")
    model.eval()
    model.module.cluster_loss_on()
    model.module.encoder_compatness()
    data_loader.sampler.set_epoch(54)
    auc = predict(model, recon_loss, data_loader, dataset)
    if auc > auc_recode:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint' + '.pth'))
    return auc


def predict(model, recon_loss, data_loader, dataset):
    scene_dict = {}
    scene_label = {}
    auc = 0
    for it, (images, idx, label, scene_num) in enumerate(data_loader):
        # move images to gpu

        predict_label = []
        truth_label = []
        index = 0

        print(str(it) + '/' + str(len(dataset)))  # 打印进度信息
        while index + args.frame_num < len(images[0, 0, :]):
            clip = images[:, :, index: index + args.frame_num]
            label_clip = label[:, index + args.frame_num]
            index = index + 1
            clip = clip.cuda()
            # images = rearrange(images, 'B C D H W -> B D C H W')
            recon, cluster_loss, space_loss, cluster_assgin, sapce_assgin = model(clip)
            # utils.save_tensor_video(recon-clip, output_dir='test_video_show')
            # utils.save_tensor_video(recon)

            # utils.tensor_medfit(recon[:, :, -1:])
            # utils.save_tensor_video(recon[:, :, 0:] - clip[:, :, 0:], output_dir="video_show_origin", save_name=str('%03d' % (index + args.frame_num))+'.jpg')
            # cluster_loss_ = torch.mean(cluster_loss).detach().cpu()
            # space_loss_ = torch.mean(space_loss).detach().cpu()
            true_video = clip[:, :, 0]
            # true_video = clip
            # true_video = true_video.squeeze()

            # cluster_loss = torch.mean(cluster_loss, [2, 3, 4])
            loss = recon_loss(recon[:, :, 0], true_video)
            # loss = utils.tensor_medfit(recon[:, :, 0] - true_video)
            loss_frame = torch.mean(loss)
            loss_frame = [loss_frame.tolist()]
            kk = utils.psnr(loss_frame)[0]
            # kk2 =cluster_loss_  + space_loss_
            psnr_frame = [kk]

            predict_label.extend(psnr_frame)
            label_clip = label_clip.tolist()[0]
            truth_label.extend([label_clip])

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
        temp = roc_auc_score(scene_label[key], scene_dict[key])
        # label_record = scene_label[key]
        # print(str(key) + "场景下的auc值为:" + str(temp))
        # plt.title(str(key))
        # plt.plot(scene_dict[key])
        # plt.show()

        auc = auc + temp

    # auc.append(roc_auc_score(truth_label, predict_label))
    # print(roc_auc_score(truth_label, predict_label))
    auc = auc / (idx + 1)
    print('AUC值为' + str(auc))
    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
