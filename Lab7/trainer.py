#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import wandb

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from dataset import get_dataset
from model import Generator, Discriminator
from dataset.iCLEVR.evaluator import evaluation_model

class Trainer():
    def __init__(self,args):
        self.args = args

        # wandb
        self.run = wandb.init(project="DLP_homework7", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.hidden_size = args.hidden_size

        # data
        trainset = get_dataset("iCLEVR", "train", "CGAN")
        testset = get_dataset("iCLEVR", args.dataset, "CGAN")
        
        # file name 
        self.file_name ="Lab7_CGAN"

        #crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        #crate folder
        self.img_path = os.path.join(args.save_folder,"predict_result")
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
            
        # dataloader
        self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4)
        self.testloader = DataLoader(dataset=testset, batch_size=32, 
        shuffle=False, num_workers=4)

        # pararmeter
        n_class = 24
        img_size = (3,64,64)

        # model 
        self.generator = Generator(args.hidden_size, n_class)
        self.discriminator = Discriminator(n_class, img_size)

        # init weight
        self.generator.weight_init(0, 0.02)
        self.discriminator.weight_init(0, 0.02)

        # show model parameter
        print(self.generator)
        print(self.discriminator)

        # wandb.watch(self.model)

        # optimizer
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.learning_rate, betas=(0.5,0.99))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(0.5,0.99))

        # criterion
        self.dis_criterion = nn.BCELoss()
        self.aux_criterion = nn.BCELoss()

        # using cuda
        if args.cuda: 
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.dis_criterion = self.dis_criterion.cuda()
            self.aux_criterion = self.aux_criterion.cuda()
            print("using cuda")

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.z_fixed = torch.randn(32, self.args.hidden_size).cuda() if self.args.cuda \
            else torch.randn(32, self.args.hidden_size)
            
        # load model if need 
        if(args.load_model != None):
            artifact = self.run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if "generator" in f:
                    self.generator.load_state_dict(torch.load(os.path.join(artifact_dir, f)))
                elif "discriminator" in f:
                    self.discriminator.load_state_dict(torch.load(os.path.join(artifact_dir, f)))

        self.evaluate(0) if args.test else self.training()

    # training
    def training(self):
        
        best_g_weight = None
        best_d_weight = None
        best_score = 0.0

        # wandb.watch(self.generator)
        # wandb.watch(self.discriminator)

        for e in range(1, self.args.epochs + 1):

            total_g = 0.0
            total_d = 0.0

            tbar = tqdm(self.trainloader)
            self.generator.train()
            self.discriminator.train()
            for i, (img, label) in enumerate(tbar):

                batch_size = label.shape[0]

                real = torch.ones(batch_size)
                fake = torch.zeros(batch_size)

                # img (batch * 3 * 64 * 64) and label (batch_size * 24)
                img, label = Variable(img),Variable(label)

                # using cuda
                if self.args.cuda:
                    img, label = img.cuda(), label.cuda()
                    real, fake = real.cuda(), fake.cuda()
                
                # train discriminator
                self.optimizer_d.zero_grad()

                label = label.float()

                # for real
                classes, predict = self.discriminator(img, label)
                loss_real = self.dis_criterion(predict, real)
                loss_real_class = self.aux_criterion(classes, label)
                
                # for fake
                z = torch.randn(batch_size, self.args.hidden_size).cuda() if self.args.cuda \
                    else torch.randn(batch_size, self.args.hidden_size)
                generated_img = self.generator(z, label)
                classes, predict = self.discriminator(generated_img.detach(), label)
                loss_fake = self.dis_criterion(predict, fake)
                loss_fake_class = self.aux_criterion(classes, label)

                # update 
                loss_d = loss_fake + loss_real + loss_real_class * 3 + loss_fake_class * 3
                loss_d.backward()
                self.optimizer_d.step()
                total_d += loss_d.item()

                # train generator
                for _ in range(5):
                    self.optimizer_g.zero_grad()

                    z = torch.randn(batch_size, self.args.hidden_size).cuda() if self.args.cuda \
                        else torch.randn(batch_size, self.args.hidden_size)
                    generated_img = self.generator(z, label)
                    classes, predict = self.discriminator(generated_img, label)
                    loss_dis = self.dis_criterion(predict, real)
                    loss_aux = self.aux_criterion(classes, label)

                    loss_g = loss_dis + loss_aux

                    # update
                    loss_g.backward()
                    self.optimizer_g.step()

                total_g += loss_g.item()

                # show loss               
                tbar.set_description('Generator loss: {0:.4f}, Discriminator loss: {1:.4f}' \
                    .format(total_g / (i + 1), total_d / (i + 1)))

            # evaluate
            score = self.evaluate(e)

            # record
            wandb.log({"score": score})
            wandb.log({"Generator loss": total_g / (i + 1)})
            wandb.log({"Discriminator loss": total_d / (i + 1)})

            # store best model
            if(score > best_score):
                best_score = score
                best_g_weight = copy.deepcopy(self.generator.state_dict())
                best_d_weight = copy.deepcopy(self.discriminator.state_dict())

        # save the best model 
        torch.save(best_g_weight, os.path.join(self.weight_path, self.file_name + "_generator" + '.pkl'))
        torch.save(best_d_weight, os.path.join(self.weight_path, self.file_name + "_discriminator" + '.pkl'))

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + "_generator" + '.pkl'))
        artifact.add_file(os.path.join(self.weight_path, self.file_name + "_discriminator" + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

    # evaluation
    def evaluate(self, e):

        tbar = tqdm(self.testloader)
        self.generator.eval()
        self.discriminator.eval()
        eval_model = evaluation_model()

        score = 0.0

        for i, label in enumerate(tbar):
            label = Variable(label)

            batch_size = label.shape[0]
                            
            # using cuda
            if self.args.cuda:
                label = label.cuda()

            label = label.float()
                
            with torch.no_grad():

                generated_img = self.generator(self.z_fixed, label)
                
            score += eval_model.eval(generated_img, label)
            tbar.set_description('Score: {0:.4f} '.format(score / (i + 1)))

            self.stored_image(generated_img, self.img_path, e)

        return score

    def stored_image(self, img, path, e):

        grid = make_grid(img, nrow=8, normalize=True)
        save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_{0}_result.png".format(e)))
        wandb.log({"generated_picture": wandb.Image(os.path.join(self.img_path, self.file_name + "_{0}_result.png".format(e)))})