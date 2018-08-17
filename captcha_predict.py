# -*- coding: UTF-8 -*-

import os
from urllib.request import urlopen
import PIL.ImageFile as ImageFile
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

import captcha_setting
from captcha_cnn_model import CNN

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

cnn = CNN()
cnn.load_state_dict(torch.load('model.pkl'))


def predict_image(image):
    """
    预测image
    :param image:
    :return:
    """
    image = image.resize((captcha_setting.IMAGE_WIDTH, captcha_setting.IMAGE_HEIGHT))
    image = transform(image)
    vimage = Variable(image[np.newaxis, :])
    predict_label = cnn(vimage)
    char_set_len = captcha_setting.ALL_CHAR_SET_LEN
    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:char_set_len].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, char_set_len:2 * char_set_len].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * char_set_len:3 * char_set_len].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * char_set_len:4 * char_set_len].data.numpy())]
    predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
    return predict_label


# if __name__ == '__main__':
#     import time
#     print("测试开始")
#     folder = captcha_setting.TRAIN_DATASET_PATH
#     start_time = time.time()
#     train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
#     total = len(train_image_file_paths)
#     correct = 0
#     for path in train_image_file_paths:
#         if predict_image(Image.open(path)) == path[-8:-4]:
#             correct = correct + 1
#     print("数量%d 正确:%d 准确率:%.2f 用时%.2fs" % (total, correct, correct * 100 / total, time.time() - start_time))


def get_image_from_url(url):
    """
    从网络读取图片
    :param url:
    :return:
    """
    fp = urlopen(url)
    p = ImageFile.Parser()  # 定义图像IO
    while 1:  # 开始图像读取
        s = fp.read(1024)
        if not s:
            break
        p.feed(s)
    image = p.close()
    return image


def predict_from_url(url):
    captcha = predict_image(get_image_from_url(url))
    return captcha


def predict_from_path(path):
    captcha = predict_image(Image.open(path))
    return captcha
