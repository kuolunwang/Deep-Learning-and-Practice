#!/usr/bin/env python3

import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, activate_function):
        super(DeepConvNet, self).__init__()

        activate = {
            'ReLU' : nn.ReLU(),
            'LeakyReLU' : nn.LeakyReLU(),
            'ELU' : nn.ELU()
        }

        # network parameter
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1,5), stride = (1,2), bias = True),
            nn.Conv2d(25, 25, kernel_size = (2,1), bias = True),
            nn.BatchNorm2d(25, affine = True),
            activate[activate_function],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1,5), stride = (1,2), bias = True),
            nn.BatchNorm2d(50, affine = True),
            activate[activate_function],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size = (1,5), bias = True),
            nn.BatchNorm2d(100, affine = True),
            activate[activate_function],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1,5), bias = True),
            nn.BatchNorm2d(200, affine = True),
            activate[activate_function],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.classify = nn.Sequential(
            nn.Linear(1600, 2, bias = True)
        )
    
    def forward(self,x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        output = x.view(x.size(0), -1)
        return self.classify(output)
