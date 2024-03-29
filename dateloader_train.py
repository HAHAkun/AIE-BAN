import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = cv2.imread(data_lowlight_path)

        data_lowlight = cv2.resize(data_lowlight, (self.size, self.size))



        data_HEB = cv2.equalizeHist(data_lowlight[:,:,0])
        data_HEG = cv2.equalizeHist(data_lowlight[:,:,1])
        data_HER = cv2.equalizeHist(data_lowlight[:,:,2])
        data_HEB = data_HEB[:, :, np.newaxis]
        data_HEG = data_HEG[:, :, np.newaxis]
        data_HER = data_HER[:, :, np.newaxis]
        data_HE = np.concatenate([data_HEB, data_HEG, data_HER], 2)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_HE = (np.asarray(data_HE) / 255.0)
        data_HE = torch.from_numpy(data_HE).float()
        return data_lowlight.permute(2, 0, 1), data_HE.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

