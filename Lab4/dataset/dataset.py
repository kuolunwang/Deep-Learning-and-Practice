#!/usr/bin/env python3
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import gdown
from zipfile import ZipFile
import shutil

class RetinopathyDataset(Dataset):
    def __init__(self, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        
        self.__downloaddata()
        self.img_name, self.label = self.__getData(mode)
        self.trans = self.__trans(mode)
        self.weight_loss = self.__cal_we_loss()
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""

        return len(self.img_name)

    def __getitem__(self, index):

        # read rgb images
        rgb_image = Image.open(os.path.join(self.root, self.img_name[index] + ".jpeg")).convert('RGB')

        # convert images
        img = self.trans(rgb_image)
        label = self.label[index]

        return img, label

    def __cal_we_loss(self):

        weight = [0 for x in range(len(set(self.label)))]
        for i in range(len(weight)):
            weight[i] = self.label[self.label == i].size

        return np.true_divide(np.max(weight), weight)


    def __trans(self, mode):
        
        if mode == "train":
            transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        else:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform


    def __downloaddata(self):

        # download dataset and extract zip to current path
        model_url = 'https://drive.google.com/u/1/uc?id=1RTmrk7Qu9IBjQYLczaYKOvXaHWBS0o72'
        model_name = 'RetinopathyDataset'

        if not os.path.isdir(os.path.join(os.getcwd(),"dataset",model_name)):
            gdown.download(model_url, output=model_name + '.zip', quiet=False)
            zip1 = ZipFile(model_name + '.zip')
            zip1.extractall(model_name)
            zip1.close()

            # move folder
            shutil.move(os.path.join(os.getcwd(),model_name), os.path.join(os.getcwd(),"dataset",model_name))

            # delete zip file
            os.remove(os.path.join(os.getcwd(),model_name + '.zip'))
        
        self.root = os.path.join(os.getcwd(), "dataset", model_name, "data")

    def __getData(self, mode):

        if mode == 'train':
            img = pd.read_csv('./dataset/train_img.csv', header = None)
            label = pd.read_csv('./dataset/train_label.csv', header = None)
            return np.squeeze(img.values), np.squeeze(label.values)
        elif mode == "test":
            img = pd.read_csv('./dataset/test_img.csv', header = None)
            label = pd.read_csv('./dataset/test_label.csv', header = None)
            return np.squeeze(img.values), np.squeeze(label.values)
        else:
            raise Excption("Error! Please input train or test")

        