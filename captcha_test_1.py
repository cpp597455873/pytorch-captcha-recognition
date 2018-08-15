# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN
import one_hot_encoding
import os


def main():
    cnn = CNN()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    test_dataloader = my_dataset.get_test1_data_loader()
    new_name_list = []
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        new_name_list.append(predict_label)

    old_file_name = [os.path.join(captcha_setting.TEST1_DATASET_PATH, image_file) for image_file in
                     os.listdir(captcha_setting.TEST1_DATASET_PATH)]
    for x in range(len(new_name_list)):
        os.rename(old_file_name[x], captcha_setting.TEST1_DATASET_PATH + new_name_list[x] + ".png")


if __name__ == '__main__':
    main()
