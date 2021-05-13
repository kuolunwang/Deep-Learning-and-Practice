#!/usr/bin/env python3

import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, activate_function):
        super(EEGNet, self).__init__()

        activate = {
            'ReLU' : nn.ReLU(),
            'LeakyReLU' : nn.LeakyReLU(),
            'ELU' : nn.ELU()
        }

        # network parameter
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1,51), stride = (1,1), padding = (0, 25), bias = False),
            nn.BatchNorm2d(16, affine = True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (2,1), stride = (1,1), groups = 16, bias = False),
            nn.BatchNorm2d(32, affine = True),
            activate[activate_function],
            nn.AvgPool2d((1, 4), stride=(1, 4), padding = 0),
            nn.Dropout(0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1, 1), padding = (0, 7), bias = False),
            nn.BatchNorm2d(32, affine = True),
            activate[activate_function],
            nn.AvgPool2d((1, 8), stride=(1, 8), padding = 0),
            nn.Dropout(0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias = True)
        )
    
    def forward(self,x):

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        output = x.view(x.size(0), -1)
        return self.classify(output)