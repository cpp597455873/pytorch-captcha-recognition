# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import captcha_setting
import os
import shutil


def random_captcha():
    captcha_text = []
    for i in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    count = 20000
    # path = captcha_setting.TRAIN_DATASET_PATH
    # path = captcha_setting.TEST_DATASET_PATH
    path = captcha_setting.TEST_DATASET_PATH
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text + '_' + now + '.png'
        image.save(path + os.path.sep + filename)
        print('saved %d : %s' % (i + 1, filename))

