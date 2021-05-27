#!/usr/bin/env python3

import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import gdown
from zipfile import ZipFile

def get_iCLEVR_data(root_folder, mode):
    
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class iCLEVRDataset(data.Dataset):
    def __init__(self, trans=None, cond=False, mode='train'):
        self.__downloaddata()
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(self.root, mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
        
        self.cond = cond
        self.num_classes = 24
        # print(self.img_list)
        # print(self.label_list)
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_list)

    def __getitem__(self, index):
        
        # read rgb images
        image = Image.open(os.path.join(self.root, "data", "images", self.img_list[index])).convert('RGB')

        # convert images
        # img = self.trans(rgb_image)
        label = self.label_list[index]

        return image, label

    def __downloaddata(self):

        # download dataset and extract zip to current path
        data_url = 'https://drive.google.com/u/1/uc?id=19_ZrzynEop2iFEaEhyG1eC-aTj4fAl_U'
        data_name = 'data'

        self.root = os.path.join(os.getcwd(), "dataset", "iCLEVR")

        if not os.path.isdir(os.path.join(self.root, data_name)):
            gdown.download(data_url, output=os.path.join(self.root, data_name + '.zip'), quiet=False)

            zip1 = ZipFile(os.path.join(self.root, data_name + '.zip'))
            zip1.extractall(os.path.join(self.root, data_name))
            zip1.close()

            os.remove(os.path.join(self.root, data_name + '.zip'))
        
        
