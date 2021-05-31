#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, hidden_size, n_class):
        super(Generator, self).__init__()

        self.embedding =  nn.Sequential(
            nn.Linear(n_class, n_class),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(n_class + hidden_size, 64, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            *self.make_block(64,128),
            *self.make_block(128,128),
            *self.make_block(128,64),
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
        c = self.embedding(c).view(-1, n_class, 1, 1)

        # concatenation z and c
        x = torch.cat((z, c), dim=1)

        # output -> batch_size * 3 * 64 * 64
        output = self.net(x)
        return output

    def make_block(self, input, output):
        block = [nn.ConvTranspose2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return block

class Discriminator(nn.Module):
    def __init__(self, n_class, img_size):
        super(Discriminator, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(n_class, int(np.prod(img_size))),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            *self.make_block(64,128),
            *self.make_block(128,128),
            *self.make_block(128,64),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # x -> batch_size * 3 * 64 * 64
        # c -> batch_size * n_class
        _, ch, w, h = x.shape

        # transfer c -> batch_size * 3 * 64 * 64
        c = self.embedding(c).view(-1, ch, w, h)

        # concatenation x and c
        x = torch.cat((x, c), dim=1)

        # output -> batch_size * 1
        output = self.net(x).view(-1)
        return output

    def make_block(self, input, output):
        block = [nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)]
        block.append(nn.BatchNorm2d(output))
        block.append(nn.LeakyReLU(0.2, inplace=True))
        return block

