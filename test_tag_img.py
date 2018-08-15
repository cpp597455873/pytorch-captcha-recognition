from lxml import html
import requests
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting
from liburl import name
import numpy as np
import captcha_setting

cnn = CNN()
cnn.load_state_dict(torch.load('model.pkl'))

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def get_one_img(image_root):
    image = Image.open(image_root)
    image = image.resize((captcha_setting.IMAGE_WIDTH, captcha_setting.IMAGE_HEIGHT))
    image = transform(image)
    return image


def predict_img(image):
    vimage = Variable(image)
    predict_label = cnn(vimage)
    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[
        np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
        predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
        predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    repredict_label = '%s%s%s%s' % (c0, c1, c2, c3)
    return repredict_label


for i in range(100):
    page = requests.get('http://readfree.me/')
    tree = html.fromstring(page.text)
    img_src = "http://readfree.me" + tree.xpath('//img[@class="captcha"]/@src')[0]
    r = requests.get(img_src)
    target_file_name = "D:\\machine_learn\\test1\\%.4d.png" % i
    with open(target_file_name, "wb") as code:
        code.write(r.content)
    img = get_one_img(target_file_name)
    plabel = predict_img(img)
    # os.rename(target_file_name, "D:\\machine_learn\\test1\\" + plabel + ".png")
