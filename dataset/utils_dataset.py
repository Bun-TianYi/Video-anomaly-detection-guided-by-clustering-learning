'''
有一个很有意思的数据集，在01_0025文件夹下的031帧开始，有人倒着走
'''
import numpy
from einops import rearrange

from misc.utils import save_tensor_video
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torch
from PIL import Image
import random
import torchvision.transforms.functional as tf
from torch import nn

rng = np.random.RandomState(2023)


def load_images(image_paths: object, transform: object) -> object:
    """
    根据给出图片序列名从内存里加载图像
    """
    imgs = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            img_str = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)

        imgs.append(Image.fromarray(img))
    if transform is not None:
        video = transform(imgs)
    # imgs = list(map(list, zip(*imgs)))      # 对列表进行以此转置，这是为了将批数据增强维度与时间维度对调，方便打包为张量

    return video


def tensor_normalize(tensor, mean=1, std=0):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    return tensor


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, frames_num=10,
                 num_pred=1, istest=False, label_folder='.', is_load_label=True, image_format='jpg', index_num=3):
        self.dir = video_folder
        self.index_num = index_num
        self.image_format = '.' + image_format
        self.istest = istest
        self.is_load_label = is_load_label
        self.transform = transform
        self.videos = OrderedDict()
        self.frames_num = frames_num
        self._num_pred = num_pred
        self.label_folder = label_folder
        self.labels = self.get_all_video_labels()
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))  # glob.glob 函数就是为了匹配符合给出路径的所有文件
        for video in sorted(videos):
            video_name = video.replace('\\', '/').split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video  # 设定一个视频帧的路径
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))  # 设定视频帧的确切名字
            self.videos[video_name]['frame'].sort()  # 对视频帧进行重排序，保证其在时间上是符合逻辑的
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])  # 设定一串视频帧的具体长度

    def get_hole_video(self, video_path):
        video_path = glob.glob(os.path.join(video_path, '*'))
        video_path.sort()
        return video_path

    def get_one_video(self, video_path, image_format, index_num=4):
        video_jpgs_num = int(video_path.split('.')[0][-index_num:])
        video_jpgs_path = video_path.split('.')[0][:-index_num]
        video_path = []
        for i in range(self.frames_num):
            num = ('%0' + str(index_num) + 'd') % (video_jpgs_num + i)
            video_path.append(video_jpgs_path + num + image_format)

        return video_path

    def get_all_video_labels(self):
        labels = []
        videos = glob.glob(os.path.join(self.label_folder, '*'))
        for video in sorted(videos):
            labels.append(video)
        return labels

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        if self.istest:
            return videos
        for video in sorted(videos):
            imgs = glob.glob(os.path.join(video, '*'))
            for i, img_path in enumerate(imgs):
                if i <= (len(imgs) - self.frames_num):
                    frames.append(img_path)
        return frames

    def __getitem__(self, index):
        video_path = self.samples[index]
        # assert len(video_path) == self.frames_num
        # frame_name = int(float(self.samples[index].replace('\\', '/').split('/')[-1].split('.')[-2]))
        if self.istest:  # 若在测试，则多传回一个标签
            # test = np.load(label_path)
            scene_num = os.path.split(video_path)[-1].split('_')[0]
            video_name = os.path.split(video_path)[-1]
            if self.is_load_label:
                video_labels = np.load(os.path.join(self.label_folder, video_name) + '.npy')
                # video_labels = np.load(self.labels[index])
                seq = self.get_hole_video(video_path)
                frames = load_images(seq, self.transform)
                frames = frames.permute(1, 0, 2, 3)
                return frames, index, video_labels, scene_num
            else:
                seq = self.get_hole_video(video_path)
                frames = load_images(seq, self.transform)
                frames = frames.permute(1, 0, 2, 3)
                return frames, index, scene_num

        seq = self.get_one_video(video_path, self.image_format, self.index_num)
        # for i in range(self._time_step + self._num_pred):
        #     image_name = self.videos[video_name]['frame'][frame_name + i]
        #     seq.append(image_name)
        frames = load_images(seq, self.transform)
        # 将矩阵由(T,C,W,H)->(C, T, W, H)
        frames = frames.permute(1, 0, 2, 3)

        return frames, index

    def __len__(self):
        return len(self.samples)


class DataTransforms(object):  # 数据转换模块
    """
    该模块可以设定视频的转换方式，当前仅设定了将视频转为张量并normalize后进行输出
    """

    def __init__(self, size=(224, 224)):
        self.tranfors = torch.nn.Sequential(
            Resize_Normalize(size, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    def __call__(self, video):
        # 这里传进来的image是plt数据结构
        video = self.tranfors(video)

        return video


class Resize_Normalize(nn.Module):
    """
    对视频进行一次标准化，并以列表形式输出，图像保存为tensor.
          """

    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.mean = mean
        self.std = std

    def forward(self, video):
        out = []
        for img in video:
            img = np.array(tf.resize(img, size=self.size)).astype(np.float32)
            # img = (img / 127.5) - 1.0
            img = img / 255
            img = torch.tensor(img)
            img = rearrange(img, 'H W C -> C H W')
            # img = (img/127.5)-1.0
            # img = tf.normalize(img, self.mean, self.std)
            out.append(img)
        return torch.stack(out)
