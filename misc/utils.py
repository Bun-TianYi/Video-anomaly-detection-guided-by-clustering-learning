import glob
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import logging
import math
import scipy.io as scio
from einops import rearrange

from PIL import Image
from skimage import io, transform, color
from torch import nn


def save_tensor_video(x, output_dir="video_show", save_name=None):
    """
    传入一个4阶（B,T,C,W,H）张量，将其变成视频保存
    """
    # (b, c, t, w, h)->(b, t, c, w, h)
    _, _, c, _, _ = x.size()
    # if c != 3:
    x = rearrange(x, 'B C D H W -> B D C H W')
    os.makedirs(output_dir, exist_ok=True)
    for i, video in enumerate(x):
        video_dir = os.path.join(output_dir, str(i))
        os.makedirs(video_dir, exist_ok=True)
        for j, image in enumerate(video):
            # max_x = torch.max(image)
            # min_x = torch.min(image)
            # image = (image - min_x) / (max_x - min_x)
            #image = (image + 1) / 2
            # image = image.detach().cpu().numpy().transpose(1, 2, 0)
            # path = os.path.join(video_dir, "img" + str(j) + ".jpg")
            # plt.imsave(path, image)
            if save_name is None:
                torchvision.utils.save_image(image,
                                             os.path.join(
                                                 video_dir, "img" + str(j) + ".jpg"))
            else:
                torchvision.utils.save_image(image, os.path.join(video_dir, save_name))

        print("第" + str(i) + "段视频保存完成")


def load_pretrain_model(ckp_path, model):
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    checkpoint = torch.load(ckp_path, map_location='cuda:0')
    checkpoint_dict = checkpoint
    # 这预训练模型文件保存的怪怪的
    model_dict = model.state_dict()
    for key, value in checkpoint_dict.items():
        # key = key.replace("backbone.", "encoder.")
        key = key[7:]
        if key in model_dict and value is not None: #and 'patchdebed' not in key: # and 'cluster' not in key:
            try:
                model_dict.update({key: value})
                print("=> loaded '{}' from checkpoint '{}'".format(key, ckp_path))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    model.load_state_dict(model_dict)
    print("模型与训练参数加载完毕")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger


def load_model(ckp_path, model):
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    checkpoint = torch.load(ckp_path)
    checkpoint_dict = checkpoint['state_dict']
    # 这预训练模型文件保存的怪怪的
    model_dict = model.state_dict()
    for key, value in checkpoint_dict.items():
        if key in model_dict and value is not None:
            try:
                model_dict.update({key: value})
                print("=> loaded '{}' from checkpoint '{}'".format(key, ckp_path))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    model.load_state_dict(model_dict)
    print("模型与训练参数加载完毕")


def psnr(mse):
    """
    计算psnr值
    """
    return [10 * math.log10(1.0 / mse_item) for mse_item in mse]


def anomly_score(psnr):
    max_psnr = max(psnr)
    min_psnr = min(psnr)

    return [1.0 - (psnr_item - min_psnr) / (max_psnr - min_psnr) for psnr_item in psnr]

# def one_zero_norm(arry):
#     return [1.0 - (psnr_item - min_psnr) / (max_psnr - min_psnr) for psnr_item in psnr]


def image_tensor2cv2(input_tensor: torch.Tensor):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    output_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    return output_tensor

def Error_thermogram_visualize(recon_image, origin_image, recon_file_name, origin_file_name, file_name):
    recon_image = image_tensor2cv2(recon_image)
    origin_image = image_tensor2cv2(origin_image)
    cv2.imwrite(recon_file_name, recon_image)
    cv2.imwrite(origin_file_name, origin_image)

    recon_image = color.rgb2gray(recon_image)
    origin_image = color.rgb2gray(origin_image)
    # 3.开始进行制作误差热力图
    A_img = origin_image
    B_img = recon_image

    # 选取需要计算差值的两幅图片
    dimg1 = A_img[:, :]
    # 归一化
    dimg1_2 = np.zeros(dimg1.shape, dtype=np.float32)
    cv2.normalize(dimg1, dimg1_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 显示选取的图像

    dimg2 = B_img[:, :]
    dimg2_2 = np.zeros(dimg2.shape, dtype=np.float32)
    cv2.normalize(dimg2, dimg2_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    # d = (abs(dimg1_2 - dimg2_2)*3)** 2
    d = (abs(dimg1_2 - dimg2_2)**2) * 10
    # d = (d - np.min(d))/(np.max(d) - np.min(d))
    fig = plt.figure(dpi=200)
    plt.figure(num=1)
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    m = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=matplotlib.cm.jet)
    m.set_array(d)
    plt.imshow(d, norm=cnorm, cmap="jet")
    plt.axis("off")
    # plt.colorbar(m)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig(file_name, bbox_inches='tight', dpi=400, pad_inches=0)
    plt.close()


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        # m.requires_grad = False


def mat2npy(save_dir):
    data_paths = glob.glob(os.path.join(save_dir, '*.mat'))
    for it, (mat_name) in enumerate(data_paths):
        data_name = mat_name.split('.')[0][-2:]
        data = scio.loadmat(mat_name)['frame_label']
        data = np.array(data)
        np.save(os.path.join(save_dir, data_name + '.npy'), data)

    print('标签抽取完毕')


def Avenue_Ped2_test_dataset_format(folder_path):
    frame_folder_path = glob.glob(os.path.join(folder_path, '*'))
    for name in frame_folder_path:
        index = int(os.path.split(name)[-1])
        newname = os.path.join(os.path.split(name)[0], '01_' + str('%04d' % index))
        os.rename(name, newname)
        print(name + "======>" + newname)


def Avenue_Ped2_test_label_format(folder_path):
    frame_folder_path = glob.glob(os.path.join(folder_path, '*.npy'))
    for name in frame_folder_path:
        index = int(os.path.split(name)[-1].split(".")[0])
        newname = os.path.join(os.path.split(name)[0], '01_' + str('%04d' % index)) + '.npy'
        os.rename(name, newname)
        print(name + "======>" + newname)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # label_color = []
    # for num in label:
    #     label_color.append(plt.cm.Set3(num))
    colors = plt.get_cmap('viridis', 5)
    label_colors = colors(range(5))
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        xs = data[i, 0]
        ys = data[i, 1]
        plt.scatter(xs, ys, color=label_colors[label[i]], s=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.show()
    # plt.pause(30)
    # l = 0
    return fig





if __name__ == '__main__':
    path = "F:/Avenue_Dataset/testing_label"
    # mat2npy(path)
    # path = "F:\Avenue_Dataset/testing_label"
    # path = "F:/Ped2/testing/frames/"
    # label_path = "F:\Ped2/test_label"
    # mat2npy(label_path)
    # Avenue_Ped2_test_label_format(label_path)
    Avenue_Ped2_test_dataset_format(path)