# 导入所有必要的库
# 该文件对ShanHaiTech数据集进行特征抽取
import glob
import cv2
import os
try:
    # 创建名为frames的文件夹
    if not os.path.exists('../frames'):
        os.makedirs('../frames')
except OSError:
    print('Error: Creating directory of data!')
dir = r'F:/ShanghaiTech/training/videos'
videos = glob.glob(os.path.join(dir, '*'))
global currentframe
currentframe = 0
for vindex in range(0, len(videos)):
    current_dir = videos[vindex]#训练集中的每一个视频
    dir_num = current_dir[-10:-3]#当前文件夹
    # if vindex != 0:
    #     pre_dir = videos[vindex-1]
    #     pre_num = pre_dir[-10:-3]
    #     if int(dir_num)-int(pre_num) == 1:
    currentframe = 0
    cam = cv2.VideoCapture(current_dir)#读取
    try:
        if not os.path.exists('../frames/' + dir_num):
            os.makedirs('../frames/' + dir_num)
    except OSError:
        print('Error: Creating directory of data!')
    times = 0
    while (True):
        ret, frame = cam.read()
        if frame is None:
            break
        # 如果视频仍然存在，继续创建图像
        if times % 18 == 0:     # 一秒采一张图
            name = '../frames/'+ dir_num +'/'+ str('%03d' % currentframe) + '.jpg'
            print('Creating...' + name)
            # 写入提取的图像
            cv2.imwrite(name, frame)
            currentframe += 1
        times += 1
