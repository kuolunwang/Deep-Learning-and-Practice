#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP homework 5", description='This lab will implement cvae')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for the trainer, default is 0.05")
        parser.add_argument("--epochs", type=int, default=500, help="The number of the training, default is 500")
        parser.add_argument("--batch_size", type=int, default=4, help="Input batch size for training, default is 4")
        parser.add_argument("--hidden_size", type=int, default=256, help="Input hidden size(256, 512) for model, default is 256")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/DLP_homework5/model:v0', default is None")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")

        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')

        # network parameter
        parser.add_argument("--threshold", type=int, default=100, help="kld loss weight parameter, default is 100")
        parser.add_argument("--kld_loss_type", type=str, default="monotonic", help="kld loss weight parameter, default is monotonic")
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 