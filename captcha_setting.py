# -*- coding: UTF-8 -*-
import os

# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

ALL_CHAR_SET = ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
IMAGE_HEIGHT = 36
IMAGE_WIDTH = 80

# TRAIN_DATASET_PATH = os.getcwd() + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep
# # TEST_DATASET_PATH = os.getcwd() + os.path.sep + 'dataset' + os.path.sep + 'test' + os.path.sep

TRAIN_DATASET_PATH = "D:\\machine_learn\\train1\\"
# TEST_DATASET_PATH = "D:\\machine_learn\\test\\"
TEST_DATASET_PATH = "D:\\machine_learn\\test\\"
TEST1_DATASET_PATH = "D:\\machine_learn\\test1\\"
PREDICT_DATASET_PATH = os.getcwd() + os.path.sep + 'dataset' + os.path.sep + 'predict' + os.path.sep
