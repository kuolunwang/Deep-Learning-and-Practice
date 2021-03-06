#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import wandb
import random

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from dataset import get_dataset
from model import Glow
import util
from dataset.iCLEVR.evaluator import evaluation_model

class Trainer():
    def __init__(self,args):
        self.args = args

        # wandb
        self.run = wandb.init(project="CNF", entity="kuolunwang") if args.dataset == "iCLEVR" else wandb.init(project="CNF_face", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.channels = args.num_channels
        config.level = args.num_levels
        config.step = args.num_steps

        # data
        if args.dataset == "iCLEVR":
            trainset = get_dataset(args.dataset, "train", "CNF")
            testset = get_dataset(args.dataset, "test", "CNF")
            new_testset = get_dataset(args.dataset, "new_test", "CNF")
        elif args.dataset == "CelebAHQ":
            trainset = get_dataset(args.dataset, "train")
        
        # file name 
        self.file_name ="Lab7_CNF"

        #crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        #crate folder
        self.img_path = os.path.join(args.save_folder,"predict_result")
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
            
        # dataloader
        if args.dataset == "iCLEVR":
            self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
            shuffle=True, num_workers=4)
            self.testloader = DataLoader(dataset=testset, batch_size=32, 
            shuffle=False, num_workers=4)
            self.new_testloader = DataLoader(dataset=new_testset, batch_size=32, 
            shuffle=False, num_workers=4)
            classes = 24
        elif args.dataset == "CelebAHQ":
            self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
            shuffle=True, num_workers=4)
            classes = 40

        # model 
        self.glow = Glow(num_channels=args.num_channels,
                        num_levels=args.num_levels,
                        num_steps=args.num_steps,
                        n_class=classes,
                        img_size=(3,64,64),
                        mode="sketch")

        # show model parameter
        print(self.glow)

        self.max_grad_norm = args.max_grad_norm

        # optimizer
        self.optimizer = torch.optim.Adam(self.glow.parameters(), lr=args.learning_rate)

        # criterion
        self.criterion = util.NLLLoss()

        # using cuda
        if args.cuda: 
            self.glow = self.glow.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.z = torch.randn(32, 3, 64, 64).cuda() if self.args.cuda \
            else torch.randn(32, 3, 64, 64)
            
        # load model if need 
        if(args.load_model != None):
            artifact = self.run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if args.dataset == "iCLEVR":
                    if args.testset in f:
                        self.glow.load_state_dict(torch.load(os.path.join(artifact_dir, f)))
                elif args.dataset == "CelebAHQ":
                    if "face" in f:
                        self.glow.load_state_dict(torch.load(os.path.join(artifact_dir, f)))

        if args.test:
            if args.testset == "test":
                self.evaluate(self.testloader, 0)
            elif args.testset == "new":
                self.evaluate(self.new_testloader, 0)
            elif args.dataset == "CelebAHQ":
                if args.task == 1:
                    self.face_generation()
                if args.task == 2:
                    self.interpolation()
                if args.task == 3:
                    self.manipulation()
        else:
            if args.dataset == "iCLEVR":
                self.training()
            elif args.dataset == "CelebAHQ":
                self.generator()

    # training
    def training(self):
        
        best_weight_test = None
        best_score_test = 0.0
        best_weight = None
        best_score = 0.0

        for e in range(1, self.args.epochs + 1):

            tbar = tqdm(self.trainloader)
            self.glow.train()
            loss_meter = util.AverageMeter()

            for i, (img, label) in enumerate(tbar):

                # img (batch * 3 * 64 * 64) and label (batch_size * 24)
                img, label = Variable(img),Variable(label)

                # using cuda
                if self.args.cuda:
                    img, label = img.cuda(), label.cuda()

                label = label.float()

                self.optimizer.zero_grad()

                z, sldj = self.glow(img, label, reverse=False)
                loss = self.criterion(z, sldj)
                loss_meter.update(loss.item(), img.size(0))
                loss.backward()
                if self.max_grad_norm > 0:
                    util.clip_grad_norm(self.optimizer, self.max_grad_norm)
                self.optimizer.step()
            
                # show loss               
                tbar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(img, loss_meter.avg),
                                     lr=self.optimizer.param_groups[0]['lr'])
                tbar.update(img.size(0))

            # evaluate
            test_score = self.evaluate(self.testloader ,e)
            score = self.evaluate(self.new_testloader, e)

            # record
            wandb.log({"test score": test_score})
            wandb.log({"new test score": score})
            wandb.log({"total loss": loss_meter.avg})

            # store best model
            if(score > best_score):
                best_score = score
                best_weight = copy.deepcopy(self.glow.state_dict())

            if(test_score > best_score_test):
                best_score_test = test_score
                best_weight_test = copy.deepcopy(self.glow.state_dict())

        # save the best model 
        torch.save(best_weight, os.path.join(self.weight_path, self.file_name + '_new' + '.pkl'))
        torch.save(best_weight_test, os.path.join(self.weight_path, self.file_name + '_test' + '.pkl'))

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '_new' + '.pkl'))
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '_test' + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

    # evaluation
    def evaluate(self, dataset, e):

        tbar = tqdm(dataset)
        self.glow.eval()
        eval_model = evaluation_model()

        score = 0.0

        for i, label in enumerate(tbar):
            label = Variable(label)
                            
            # using cuda
            if self.args.cuda:
                label = label.cuda()

            label = label.float()
                
            with torch.no_grad():

                generated_img, _ = self.glow(self.z, label, reverse=True)
                
            score += eval_model.eval(generated_img, label)
            tbar.set_description('Score: {0:.4f} '.format(score / (i + 1)))

            self.stored_image(generated_img, self.img_path, e, dataset)

        return score

    # save generated images
    def stored_image(self, img, path, e, dataset):

        grid = make_grid(img, nrow=8, normalize=True)
        if dataset == self.testloader:
            save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_test" + "_{0}_result.png".format(e)))
            wandb.log({"generated picture for test": wandb.Image(os.path.join(self.img_path, self.file_name + "_test" + "_{0}_result.png".format(e)))})
        elif dataset == self.new_testloader:
            save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_new_test" + "_{0}_result.png".format(e)))
            wandb.log({"generated picture for new test": wandb.Image(os.path.join(self.img_path, self.file_name + "_new_test" + "_{0}_result.png".format(e)))})

    # train generator for task2
    def generator(self):

        best_weight = None
        best_loss = 100000

        for e in range(1, self.args.epochs + 1):

            tbar = tqdm(self.trainloader)
            self.glow.train()
            loss_meter = util.AverageMeter()

            for i, (img, label) in enumerate(tbar):

                # img (batch * 3 * 64 * 64) and label (batch_size * 24)
                img, label = Variable(img),Variable(label)

                # using cuda
                if self.args.cuda:
                    img, label = img.cuda(), label.cuda()

                label = label.float()

                self.optimizer.zero_grad()

                z, sldj = self.glow(img, label, reverse=False)
                loss = self.criterion(z, sldj)
                loss_meter.update(loss.item(), img.size(0))
                loss.backward()
                if self.max_grad_norm > 0:
                    util.clip_grad_norm(self.optimizer, self.max_grad_norm)
                self.optimizer.step()
            
                # show loss               
                tbar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(img, loss_meter.avg),
                                     lr=self.optimizer.param_groups[0]['lr'])
                tbar.update(img.size(0))

            # record
            wandb.log({"total loss": loss_meter.avg})

            # store best model
            if(best_loss > loss_meter.avg):
                best_loss = loss_meter.avg
                best_weight = copy.deepcopy(self.glow.state_dict())

        # save the best model 
        torch.save(best_weight, os.path.join(self.weight_path, self.file_name + '_face' + '.pkl'))

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '_face' + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

    def face_generation(self):

        self.glow.eval()

        z = torch.randn(5, 3, 64, 64).cuda() if self.args.cuda \
            else torch.randn(5, 3, 64, 64)

        # calculate condition
        for i in range(5):
            cond = torch.tensor([1 if random.random() < 0.2 else -1 for _ in range(40)]).view(1, -1)
            label = cond if i == 0 else torch.cat((label, cond), dim=0)

        label = Variable(label)
                            
        # using cuda
        if self.args.cuda:
            label = label.cuda()
            
        with torch.no_grad():

            generated_img, _ = self.glow(z, label.float(), reverse=True)

            generated_img = torch.tanh(generated_img)
                
        grid = make_grid(generated_img, nrow=8, normalize=True)
        save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_face_generation_result.png"))
        wandb.log({"generated picture for face": wandb.Image(os.path.join(self.img_path, self.file_name + "_face_generation_result.png"))})

    def interpolation(self):

        # sample from training dataset 
        sample = next(iter(self.trainloader))
        img, con = sample

        img, con = Variable(img), Variable(con)
                            
        # using cuda
        if self.args.cuda:
            img = img.cuda()
            con = con.cuda()
        
        with torch.no_grad():

            z, _ = self.glow(img[0:4], con[0:4].float(), reverse=False)

            z_a, z_b = z[0:3], z[1:4]
            z_diff = z_b - z_a
            
            for i in range(8):
                alpha = (1.0 / 7) * i
                z_c = z_a + alpha * z_diff
                if i == 0:
                    generated_img, _ = self.glow(z_c, con[1:4].float(), reverse=True) 
                else: 
                    g, _ = self.glow(z_c, con[1:4].float(), reverse=True) 
                    generated_img = torch.cat((generated_img, g), dim=0)

            t = generated_img.view(8, 3, 3, 64, 64)
            generated_img = t.permute(1, 0, 2, 3, 4)
            generated_img = torch.tanh(generated_img.reshape(24, 3, 64, 64))

        grid = make_grid(generated_img, nrow=8, normalize=True)
        save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_interpolation_result.png"))
        wandb.log({"generated picture for interpolation": wandb.Image(os.path.join(self.img_path, self.file_name + "_interpolation_result.png"))})

    def manipulation(self):

        # sample from training dataset 
        sample = next(iter(self.trainloader))
        img, con = sample

        img, con = Variable(img), Variable(con)
                            
        # using cuda
        if self.args.cuda:
            img = img.cuda()
            con = con.cuda()

        # select attribute (smiling)
        # att = random.randint(0,40)
        att = 31
        
        with torch.no_grad():

            z, _ = self.glow(img, con.float(), reverse=False)

            a_with = con[1:,att] == 1
            a_without = con[1:,att] == -1
            index_w, index_wo = torch.nonzero(a_with).view(-1), torch.nonzero(a_without).view(-1)

            z_w, z_wo = z[index_w], z[index_wo]
            z_diff = z_w.mean(0) - z_wo.mean(0)
            
            for i in range(5):
                
                alpha = (1.0 / 4) * i
                if con[0, att] == 1:
                    z_c = z[0] - alpha * z_diff
                else:
                    z_c = z[0] + alpha * z_diff
                z_c = z_c.view(1, 3, 64, 64)
                if i == 0:
                    generated_img, _ = self.glow(z_c, con[0].float(), reverse=True) 
                else: 
                    g, _ = self.glow(z_c, con[0].float(), reverse=True) 
                    generated_img = torch.cat((generated_img, g), dim=0)

            generated_img = torch.tanh(generated_img)

        grid = make_grid(generated_img, nrow=8, normalize=True)
        save_image(grid, format="png", fp=os.path.join(self.img_path, self.file_name + "_mainpulation_result.png"))
        wandb.log({"generated picture for mainpulation": wandb.Image(os.path.join(self.img_path, self.file_name + "_mainpulation_result.png"))})


