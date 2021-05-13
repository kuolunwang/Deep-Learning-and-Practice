#!/usr/bin/env python3

import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class EEGDataset(Dataset):
    def __init__(self, mode):
        
        if(mode == "train"):
            S4b = np.load('./dataset/S4b_train.npz')
            X11b = np.load('./dataset/X11b_train.npz')
        elif(mode == "test"):
            S4b = np.load('./dataset/S4b_test.npz')
            X11b = np.load('./dataset/X11b_test.npz')
        else:
            raise Excption("Error! Please input train or test")

        data = np.concatenate((S4b['signal'], X11b['signal']), axis=0)
        label = np.concatenate((S4b['label'], X11b['label']), axis=0) - 1

        data = np.transpose(np.expand_dims(data, axis=1), (0, 1, 3, 2))

        mask = np.where(np.isnan(data))
        data[mask] = np.nanmean(data)

        self.data, self.label = torch.from_numpy(data).float(),torch.from_numpy(label).float()

    def __getitem__(self, index):

        return self.data[index], self.label[index]

    def __len__(self):

        return self.data.shape[0]