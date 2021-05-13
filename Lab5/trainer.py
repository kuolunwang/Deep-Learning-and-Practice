#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
import copy
import wandb
import random
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from dataset.dataset import WordDataset
from model.cvae import CVAE
from dataset.converter import Converter
from util import teacher_forcing_ratio, get_kld_weight

class Trainer():
    def __init__(self,args):
        self.args = args

        # wandb
        self.run = wandb.init(project="DLP_homework5", entity="kuolunwang")
        config = wandb.config

        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.hidden_size = args.hidden_size
        config.kld_loss_type = args.kld_loss_type
        config.threshold = args.threshold

        # data
        trainset = WordDataset("train")
        testset = WordDataset("test")

        # converter
        self.converter = Converter()

        # max length
        self.max_length = trainset.max_length + 1
        
        # file name 
        self.file_name ="{0}_threshold{1}_batch{2}_epochs{3}_learning rate{4}" \
            .format(args.kld_loss_type, args.threshold, args.batch_size, args.epochs, args.learning_rate)

        #crate folder
        self.weight_path = os.path.join(args.save_folder,"weight")
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
            
        # dataloader
        self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
        shuffle=True, num_workers=4, collate_fn = trainset.collate_fn)
        self.testloader = DataLoader(dataset=testset, batch_size=1, 
        shuffle=False, num_workers=4)

        # model 
        self.model = CVAE(args.hidden_size, use_cuda=self.args.cuda)

        # show model parameter
        print(self.model)

        # wandb.watch(self.model)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # using cuda
        if args.cuda: 
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")

        self.device = torch.device("cuda" if args.cuda else "cpu")
            
        # load model if need 
        if(args.load_model != None):
            artifact = self.run.use_artifact(args.load_model, type='model')
            artifact_dir = artifact.download()
            files = os.listdir(artifact_dir)
            for f in files:
                if ".pkl" in f:
                    file_name = f
            self.model.load_state_dict(torch.load(os.path.join(artifact_dir, file_name)))

        self.evaluate() if args.test else self.training()

    def loss_function(self, criterion, distribution, target, hid_mean, hid_logvar, cell_mean, cell_logvar):

        """
        distribution : batch_size * length * input_size
        target : batch_size * length
        mean : 1 * batch_size * latent_size
        logvar : 1 * batch_size * latent_size
        """
        # cross entropy loss
        batch_size, length, _ = distribution.shape

        CE = criterion(distribution.permute(0, 2, 1), target[:,:length])
        
        # kl loss
        KLD_hid = -0.5 * torch.sum(1 + hid_logvar - hid_mean.pow(2) - hid_logvar.exp())
        KLD_cell = -0.5 * torch.sum(1 + cell_logvar - cell_mean.pow(2) - cell_logvar.exp())

        return CE, (KLD_hid + KLD_cell) / batch_size

    def get_gaussian_score(self, words):
        words_list = []
        score = 0
        
        yourpath = os.path.join(self.args.save_folder, "dataset", "train.txt")
        with open(yourpath,'r') as fp:
            for line in fp:
                word = line.strip('\n').split(' ')
                words_list.extend(word)
            for i in set(words):
                if(i in words_list):
                    score += 1
        return score/len(words)

    def compute_bleu(self, output, reference):
        cc = SmoothingFunction()
        if len(reference) == 3:
            weights = (0.33,0.33,0.33)
        else:
            weights = (0.25,0.25,0.25,0.25)
        return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

    # training
    def training(self):
        
        best_model_weight = None
        best_bleu = 0.0

        # wandb.watch(self.model)

        for e in range(1, self.args.epochs + 1):

            total_loss = 0.0
            CE_loss = 0.0
            KLD_loss = 0.0
            BLEU_score = 0.0

            tbar = tqdm(self.trainloader)
            self.model.train()
            for i, (word, condition) in enumerate(tbar):

                # word (batch_size * length), condition (batch_size)
                word, condition = Variable(word),Variable(condition)

                # using cuda
                if self.args.cuda:
                    word, condition = word.cuda(), condition.cuda()

                # teacher forcing and kld wieght                
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio(e, self.args.epochs) else False
                kld_weight = get_kld_weight(self.args.kld_loss_type, self.args.epochs, e, self.args.threshold)

                # predict
                prediction, distribution, hid_mean, hid_logvar, cell_mean, cell_logvar = self.model(word, condition, use_teacher_forcing)

                # calculate loss
                CE, KLD = self.loss_function(self.criterion, distribution, word, hid_mean, hid_logvar, cell_mean, cell_logvar)
                loss = CE + kld_weight * KLD
                CE_loss += CE.item()
                KLD_loss += KLD.item()
                for k in range(prediction.shape[0]):
                    predict, target = self.converter.tensor2word(prediction[k]), self.converter.tensor2word(word[k])
                    BLEU_score += self.compute_bleu(predict, target)

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # show loss               
                tbar.set_description('Total loss: {0:.4f}, CE loss: {1:.4f}, KLD loss: {2:.4f}, BLEU score: {3:.4f}' \
                    .format(total_loss / (i + 1), CE_loss / (i + 1), KLD_loss / (i + 1), BLEU_score / (prediction.shape[0] * (i + 1))))

            # evaluate
            test_predict, bleu_score = self.evaluate()

            # record
            wandb.log({"teacher forcing ratio": teacher_forcing_ratio(e, self.args.epochs)})
            wandb.log({"kld weight": kld_weight})
            wandb.log({"total loss": total_loss / (i + 1)})
            wandb.log({"CE loss": CE_loss / (i +1)})
            wandb.log({"KLD loss": KLD_loss / (i + 1)})
            wandb.log({"Train BLEU score": BLEU_score / (prediction.shape[0] * (i + 1))})
            wandb.log({"Test BLEU score": bleu_score})

            print(test_predict)

            # store best model
            if(bleu_score > best_bleu):
                best_bleu = bleu_score
                best_model_weight = copy.deepcopy(self.model.state_dict())

        # save the best model 
        torch.save(best_model_weight, os.path.join(self.weight_path, self.file_name + '.pkl'))

        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(os.path.join(self.weight_path, self.file_name + '.pkl'))
        self.run.log_artifact(artifact)
        self.run.join()

    # evaluation
    def evaluate(self):

        tbar = tqdm(self.testloader)
        self.model.eval()
        BLEU_score = 0.0
        predict_list = []

        for i, (input_word, input_tense, target_word, target_tense) in enumerate(tbar):
            input_word, input_tense = Variable(input_word),Variable(input_tense)
            target_word, target_tense = Variable(target_word),Variable(target_tense)
                            
            # using cuda
            if self.args.cuda:
                input_word, input_tense = input_word.cuda(), input_tense.cuda()
                target_word, target_tense = target_word.cuda(), target_tense.cuda()
                
            with torch.no_grad():

                prediction = self.model.inference(input_word, input_tense, target_tense, self.max_length)
                predict, target = self.converter.tensor2word(prediction[0]), self.converter.tensor2word(target_word[0])
                BLEU_score += self.compute_bleu(predict, target)
                predict_list.append([predict, target, self.converter.tensor2word(input_word[0])])

            tbar.set_description('Test BLEU score: {0:.4f} '.format(BLEU_score/ (i + 1)))
            
        return predict_list, BLEU_score / (i + 1)

    def Gaussian_score(self):

        self.model.eval()
        predict_list = []
        tense = torch.tensor([i for i in range(4)]).to(self.device)

        with torch.no_grad():
            for i in range(100):
                latent = torch.randn(1, 1, 32).to(self.device).expand(1,4,32)
                cell = torch.randn(1, 1, 32).to(self.device).expand(1,4,32)
                predict_word = self.model.generate(tense, latent, cell, self.max_length)
                predict_list.extend([self.converter.tensor2word(predict_word[i]) for i in range(4)])
        print(predict_list)

        print("Gaussian score : {0:.2f}".format(self.get_gaussian_score(predict_list)))
