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

def get_CelebA_data(root_folder):
    
    img_list = os.listdir(os.path.join(root_folder, "data", 'images'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list

class CelebAHQDataset(data.Dataset):
    def __init__(self, mode=None):
        self.__downloaddata()
        self.trans = self.__trans()
        self.img_list, self.label_list = get_CelebA_data(self.root)
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_list)

    def __getitem__(self, index):
        
        # read rgb images
        image = Image.open(os.path.join(self.root, "data", "images", self.img_list[index])).convert('RGB')

        # convert images
        img = self.trans(image)

        label = self.label_list[index]

        return img, torch.tensor(label)

    def __trans(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64,64))
        ])

        return transform

    def __downloaddata(self):

        # download dataset and extract zip to current path
        data_url = 'https://drive.google.com/u/1/uc?id=1phoYB2JVZ9DVMfRDX22LlYwlOxARv3Qe'
        data_name = 'data'

        self.root = os.path.join(os.getcwd(), "dataset", "CelebAHQ")

        if not os.path.isdir(os.path.join(self.root, data_name)):
            gdown.download(data_url, output=os.path.join(self.root, data_name + '.zip'), quiet=False)

            zip1 = ZipFile(os.path.join(self.root, data_name + '.zip'))
            zip1.extractall(os.path.join(self.root, data_name))
            zip1.close()

            os.remove(os.path.join(self.root, data_name + '.zip'))

        