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
from sklearn.manifold import TSNE
from torchvision import models as torchvision_models
import dataset.utils_dataset as dt
from misc import utils
from utils import distritributed_model as dp
from model import backbone
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('Swin Transformer', add_help=False)

    # Model parameters
    parser.add_argument('--frame_num', default=2, type=int, help="请输入一次读取的视频帧数")
    parser.add_argument('--patch_size', default=(2, 4, 4), type=tuple, help="请输入patch编码的大小")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    # parser.add_argument('--data_path', default='F:/ShanghaiTech/testing/test_frames', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:\ShanghaiTech/testing/test_frame_mask', type=str)
    # parser.add_argument('--data_path', default='F:/Avenue_Dataset/test', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    # parser.add_argument('--label_path', default='F:/Avenue_Dataset/testing_label', type=str)
    parser.add_argument('--data_path', default='F:/ShanghaiTech/testing/test_frames', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--label_path', default='F:\ShanghaiTech/testing/test_frame_mask', type=str)



    parser.add_argument('--model_pretrain', default='F:\My-repository\My_Model\log_dir\checkpoint18.pth',
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
    model = backbone.Mymodel(args, ispredict=False, iscluster=False)
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
    record = {}
    label_num = np.ones(1024)
    for it, (images, idx, num_scene) in enumerate(data_loader):
        # move images to gpu

        index = 0

        print(str(it) + '/' + str(len(dataset)))  # 打印进度信息
        while index + args.frame_num < len(images[0, 0, :]):
            clip = images[:, :, index: index + args.frame_num]
            index = index + 1
            clip = clip.cuda()
            recon, cluster_loss, space_loss, cluster_assgin, sapce_assgin, feature, feature_label = model(clip)

            for data, label in zip(feature, feature_label):
                label = int(label.item())
                data = data.detach().cpu().numpy()
                if label in record.keys():
                    temp = record[label]
                    record[label] = np.vstack((temp, data))
                    label_num[label] = label_num[label] + 1
                else:
                    record.update({label: data})

        label_max = np.flip(np.argsort(label_num, )[[-5, -4, -3, -6]])
        data = record[label_max[0]]
        # data = record[0]
        # data = data[0:8000]
        # label = np.zeros(data.shape[0], dtype='int')
        tag = 1
        label = np.ones(data.shape[0], dtype='int') * tag
        ll = label[0]
        for num in label_max[1:]:
            tag = tag + 1
            temp = record[num]
            temp_label = np.ones(temp.shape[0], dtype='int') * tag
            label = np.concatenate((label, temp_label))
            data = np.vstack((data, temp))
        tsne = TSNE(n_components=2, init='pca', random_state=1, learning_rate=1000)
        result = tsne.fit_transform(data)
        fig = utils.plot_embedding(result, label, title='tsne')
        plt.savefig('C:/Users/user/Desktop/基于视频理解的异常检测论文/论文图片/特征聚类.jpg'
                    , bbox_inches='tight', dpi=400, pad_inches=0)





if __name__ == '__main__':
    # 这里需要对数据加强模块进行一次大改动，需要将逐像素翻转变成对整个视频段的处理
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_error(args)

