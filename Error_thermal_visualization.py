from PIL import Image
import cv2
import matplotlib
from skimage import io, transform, color
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import dataset.utils_dataset as dt
from misc import utils
from utils import distritributed_model as dp
from model import backbone
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Swin Transformer', add_help=False)

    # Model parameters
    parser.add_argument('--frame_num', default=5, type=int, help="请输入一次读取的视频帧数")
    parser.add_argument('--patch_size', default=(2, 4, 4), type=tuple, help="请输入patch编码的大小")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    # parser.add_argument('--data_path', default='F:/ShanghaiTech/testing/test_frames', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:\ShanghaiTech/testing/test_frame_mask', type=str)
    # parser.add_argument('--data_path', default='F:/Avenue_Dataset/test', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:/Avenue_Dataset/testing_label', type=str)
    parser.add_argument('--data_path', default='F:\Ped2/test/frames', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--label_path', default='F:\Ped2/test_label', type=str)



    parser.add_argument('--model_pretrain', default='F:\My-repository\My_Model\log_dir\checkpoint69.pth',
                        type=str, help='训练好的模型路径')
    parser.add_argument('--output_dir', default="error_visualize", type=str, help='保存输出文件路径')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser




def main_error(args):
    dp.init_distributed_mode(args)
    dp.fix_random_seeds(0)
    print("git:\n  {}\n".format(dp.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ========准备数据集========#
    transform = dt.DataTransforms()
    dataset = dt.DataLoader(args.data_path, transform=transform,
                            frames_num=args.frame_num, label_folder=args.label_path,
                            istest=True, is_load_label=False)
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
    model = backbone.Mymodel(args, ispredict=True)
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
    model.eval()
    model.module.cluster_loss_on()
    model.module.encoder_compatness()
    data_loader.sampler.set_epoch(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists("true_visualize"):
        os.makedirs("true_visualize")

    if not os.path.exists("recon_visualize"):
        os.makedirs("recon_visualize")

    data_iter = predict(model, recon_loss, data_loader, dataset, data_iter)


def predict(model, recon_loss, data_loader, dataset, data_iter):
    for it, (images, idx, num_scene) in enumerate(data_loader):
        # move images to gpu

        predict_label = []
        index = 0

        print(str(it) + '/' + str(len(dataset)))  # 打印进度信息
        while index + args.frame_num < len(images[0, 0, :]):
            clip = images[:, :, index: index + args.frame_num]

            index = index + 1
            true_video = clip[:, :, -1].cuda()
            clip = clip[:, :, 0:4].cuda()
            recon, cluster_loss, space_loss, cluster_assgin, sapce_assgin = model(clip)

            recon_video = recon[:, :, 0]
            true_file_name = os.path.join("true_visualize", str('%03d' % (index + args.frame_num))+'.png')
            recon_file_name = os.path.join("recon_visualize", str('%03d' % (index + args.frame_num))+'.png')
            filename = os.path.join(args.output_dir, str('%03d' % (index + args.frame_num))+'.png')
            utils.Error_thermogram_visualize(recon_video, true_video, recon_file_name, true_file_name, filename)


if __name__ == '__main__':
    # 这里需要对数据加强模块进行一次大改动，需要将逐像素翻转变成对整个视频段的处理
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_error(args)

