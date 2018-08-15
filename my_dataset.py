# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting


class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform
        self.iw = captcha_setting.IMAGE_WIDTH
        self.ih = captcha_setting.IMAGE_HEIGHT

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        image = image.resize((self.iw, self.ih))
        # print("width" + str(image.width) + " height" + str(image.height))
        if self.transform is not None:
            image = self.transform(image)
        name_ = image_name[0:4]
        label = ohe.encode(name_)  # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字_时间戳.PNG", 4个数字即是图片的验证码的值,同时对该值做 one-hot 处理
        return image, label


transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_train_data_loader():
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = mydataset(captcha_setting.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_test1_data_loader():
    dataset = mydataset(captcha_setting.TEST1_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def get_predict_data_loader():
    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
