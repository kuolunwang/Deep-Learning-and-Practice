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

from dataset.dataset import EEGDataset
from model import get_model

class Trainer():
    def __init__(self,args):
        self.args = args

        # wandb
        self.run = wandb.init(project="DLP_homework3", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.epochs = args.epochs

        # data
        trainset = EEGDataset("train")
        testset = EEGDataset("test")
        
        # file name 
        self.file_name ="{0}_{1}_{2}_{3}_{4}".format(args.network, args.activate_function, args.learning_rate, args.epochs, args.batch_size)

        # crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

        self.picture_path = os.path.join(args.save_folder,"picture")
        if not os.path.exists(self.picture_path):
            os.makedirs(self.picture_path)

        self.record_path = os.path.join(args.save_folder,"record")
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
            
        # dataloader
        self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2)
        self.testloader = DataLoader(dataset=testset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2)

        # model 
        self.model = get_model(args.network, args.activate_function)

        # show model parameter
        print(self.model)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # using cuda
        if args.cuda: 
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")
            
        # load model if need 
        if(args.load_model != None):
            artifact = self.run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if ".pkl" in f:
                    file_name = f
            self.model.load_state_dict(torch.load(os.path.join(artifact_dir, file_name)))

        self.evaluate(self.testloader) if args.test else self.training()

    # training
    def training(self):
        
        best_model_weight = None
        best_accuracy = 0.0
        self.loss_record = []
        wandb.watch(self.model)
        for e in range(self.args.epochs):
            train_loss = 0.0
            tbar = tqdm(self.trainloader)
            self.model.train()
            for i, (data, label) in enumerate(tbar):
                data, label = Variable(data),Variable(label)

                # using cuda
                if self.args.cuda:
                    data, label = data.cuda(), label.cuda()

                prediction = self.model(data)
                loss = self.criterion(prediction, label.long())

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.optimizer.zero_grad()

                tbar.set_description('Train loss: {0:.6f}'.format(train_loss / (i + 1)))

            self.loss_record.append(train_loss / (i + 1))
            tr_ac = self.evaluate(self.trainloader)
            te_ac = self.evaluate(self.testloader)
            wandb.log({"loss":train_loss / (i + 1)})
            wandb.log({"Test accuracy": te_ac})
            wandb.log({"Train accuracy": tr_ac})
            self.record(e+1, train_loss / (i + 1), tr_ac, te_ac)

        if(te_ac > best_accuracy):
            best_accuracy = te_ac
            best_model_weight = copy.deepcopy(self.model.state_dict())


        print('Train Accuracy: %2.2f%%' % (tr_ac))
        print('Test Accuracy: %2.2f%%' % (te_ac))

        # save the best model 
        torch.save(best_model_weight, os.path.join(self.weight_path, self.file_name + '.pkl'))

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

        # plot learning curve
        self.plt_lr_cur()

    # evaluation
    def evaluate(self, d):

        correct = 0.0
        total = 0.0
        tbar = tqdm(d)
        self.model.eval()
        for i, (data, label) in enumerate(tbar):
            data, label = Variable(data),Variable(label)
                            
            # using cuda
            if self.args.cuda:
                data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(label.data.view_as(pred))).cpu().numpy())
                total += data.size(0)

            text = "Train accuracy" if d == self.trainloader else "Test accuracy"
            tbar.set_description('{0}: {1:2.2f}% '.format(text, 100. * correct / total))

        return 100.0 * correct / total

    # plot learning curve 
    def plt_lr_cur(self):

        plt.title("learning curve", fontsize = 18)
        ep = [x for x in range(1, self.args.epochs + 1)]
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(ep, self.loss_record)
        plt.savefig(os.path.join(self.picture_path, self.file_name + '_learning_curve.png'))
        # plt.show()

    # record information
    def record(self, epochs, loss, tr_ac, te_ac):

        file_path = os.path.join(self.record_path, self.file_name + '.txt')
        with open(file_path, "a") as f:
            f.writelines("Epochs : {0}, train loss : {1:.6f}, train accurancy : {2:.2f}, test accurancy : {3:.2f}\n".format(epochs, loss, tr_ac, te_ac))