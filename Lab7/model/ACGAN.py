#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, hidden_size, n_class):
        super(Generator, self).__init__()

        self.embedding =  nn.Sequential(
            nn.Linear(n_class, 100),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(100 + hidden_size, 512, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            *self.make_block(512, 256),
            *self.make_block(256, 128),
            *self.make_block(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, c):
        # z -> batch_size * hidden_size
        # c -> batch_size * n_class
        _, hidden_size = z.shape
        _, n_class = c.shape

        # transfer z -> batch_size * hidden_size * 1 * 1
        # transfer c -> batch_size * n_class * 1 * 1
        z = z.view(-1, hidden_size, 1, 1)
        c = self.embedding(c).view(-1, 100, 1, 1)

        # concatenation z and c
        x = torch.cat((z, c), dim=1)

        # output -> batch_size * 3 * 64 * 64
        output = self.net(x)
        return output

    def make_block(self, input, output):
        block = [nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.ReLU())
        return block

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, n_class):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            *self.make_block(4, 64),
            *self.make_block(64,128),
            *self.make_block(128,256),
            *self.make_block(256,512)
        )

        self.dis = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.aux = nn.Sequential(
            nn.Linear(512, n_class),
            nn.Softmax()
        )

    def forward(self, x):
        # x -> batch_size * 3 * 64 * 64

        # output -> batch_size * 512 * 1 * 1
        output = self.net(x)

        classes = self.aux(output.view(output.shape[0], -1))
        realfake = self.dis(output).view(-1)

        return classes, realfake

    def make_block(self, input, output):
        block = [nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.LeakyReLU())
        return block

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()