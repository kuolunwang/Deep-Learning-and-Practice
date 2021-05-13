#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP homework 3", description='This lab implement EEGNet and DeepConvNet with pytorch')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the trainer, default is 1e-2")
        parser.add_argument("--epochs", type=int, default=300, help="The number of the training, default is 300")
        parser.add_argument("--batch_size", type=int, default=64, help="Input batch size for training, default is 64")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_homework3/model:v0', default is None")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")

        # cuda
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training, default is False')

        # network
        parser.add_argument("--network", type=str, default="EEGNet", help="select network EEGNet or DeepConvNet, default is EEGNet")

        # activation function
        parser.add_argument("--activate_function", type=str, default="ReLU", help="activation function (ReLU, LeakyReLU, ELU)for the network, default is ReLU")
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 