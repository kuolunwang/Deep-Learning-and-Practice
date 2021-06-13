

#!/usr/bin/env python3

import argparse
import torch
import os 

class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(prog="DLP homework 7", description='This lab will implement cgan')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the trainer, default is 0.0002")
        parser.add_argument("--epochs", type=int, default=250, help="The number of the training, default is 250")
        parser.add_argument("--batch_size", type=int, default=16, help="Input batch size for training, default is 32")
        parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
        parser.add_argument('--num_levels', '-L', default=4, type=int, help='Number of levels in the Glow model')
        parser.add_argument('--num_steps', '-K', default=6, type=int, help='Number of steps of flow in each level')
        parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
        
        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(), help="save model in save folder, default is current path")
        parser.add_argument("--load_model",  type=str, default=None, help="load model from wandb, ex. 'kuolunwang/CNF/model:v0', default is None")
        parser.add_argument("--testset",  type=str, default=None, help="evaluate test dataset(new and test), default is None")
        parser.add_argument("--dataset",  type=str, default="iCLEVR", help="This lab have two dataset, iCLEVR and CelebAHQ, default is iCLEVR")
        parser.add_argument("--task",  type=int, default=1, help="The task divide into three parts, face generation(1), linear interpolation(2) and attribute manipulation(3), default is 1")
        parser.add_argument("--test", action="store_true", default=False, help="True is test model, False is keep train, default is False")


        # cuda
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training, default is False')
        
        self.parser = parser

    def create(self):

        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args 