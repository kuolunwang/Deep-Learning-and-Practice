#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, hidden_size, use_cuda, input_size=29, latent_size=32, tense=4, condition_size=8):
        super(CVAE, self).__init__()
        self.condition_size = condition_size
        self.hidden_size = hidden_size
        self.te_em = nn.Embedding(tense, condition_size)
        self.encoder = Encoder(input_size, hidden_size, use_cuda)
        self.decoder = Decoder(input_size, hidden_size, use_cuda)
        self.hid2mean = nn.Linear(hidden_size, latent_size)
        self.hid2logvar = nn.Linear(hidden_size, latent_size)
        self.cell2mean = nn.Linear(hidden_size, latent_size)
        self.cell2logvar = nn.Linear(hidden_size, latent_size)
        self.lat2hid = nn.Linear(latent_size, hidden_size - condition_size)
        self.lat2cell =  nn.Linear(latent_size, hidden_size - condition_size)
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, input, condition, use_teacher_forcing):
        
        # get batch_size and length
        batch_size, length = input.shape

        # init encoder hidden and cell state (1 * batch_size * (hidden_size - condition_size) )
        en_hidden = self.encoder.inithidden(batch_size, self.hidden_size - self.condition_size)
        en_cell_state = self.encoder.initcell(batch_size, self.hidden_size - self.condition_size)

        # tense embedding (batch_size * 1 * condition_size) -> (1 * batch_size  * condition_size)
        tense_embedding = self.te_em(condition.view(-1, 1))
        tense_embedding = tense_embedding.permute(1, 0, 2)

        # cat condition into hidden and cell state
        en_hidden = torch.cat((en_hidden, tense_embedding), dim=2)
        en_cell_state = torch.cat((en_cell_state, tense_embedding), dim=2)
        # en_cell_state = torch.zeros_like(en_hidden, device=self.device)

        # encoder lstm
        for i in range(length):
            _, en_hidden, en_cell_state = self.encoder(input[:,i].view(-1,1), en_hidden, en_cell_state)

        # fully connection extract mean and logvar (1 * batch_size * latent_size)
        hid_mean = self.hid2mean(en_hidden)
        hid_logvar = self.hid2logvar(en_hidden)
        cell_mean = self.cell2mean(en_cell_state)
        cell_logvar = self.cell2logvar(en_cell_state)

        # reparameterize (1 * batch_size * latent_size)
        hid_latent = self.reparameterize(hid_mean, hid_logvar)
        cell_latent = self.reparameterize(cell_mean, cell_logvar)

        # decoder hidden and cell state (1 * batch_size * hidden_size)
        de_hidden = self.lat2hid(hid_latent)
        de_cell_state = self.lat2cell(cell_latent)
        de_hidden = torch.cat((de_hidden, tense_embedding), dim=2)
        de_cell_state = torch.cat((de_cell_state, tense_embedding), dim=2)
        # de_cell_state = torch.zeros_like(de_hidden, device=self.device)
        
        # decoder input(SOS = 0)
        de_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        # PAD tensor
        pad = torch.tensor([2 for i in range(batch_size)], device=self.device).view(-1,1)

        # decoder lstm
        for i in range(length):
            output, de_hidden, de_cell_state = self.decoder(de_input, de_hidden, de_cell_state)

            # reconstruction
            number = torch.max(output, dim=2)[1]
            predict = torch.cat((predict, number), dim=1) if i!=0 else number

            # predict distribution (batch_size * 1ength * input_size)
            distribution = torch.cat((distribution, output), dim=1) if i!=0 else output

            # teacher forcing
            if(use_teacher_forcing):
                de_input = input[:,i].view(-1,1)
            else:
                de_input = number

            # transform eos to pad
            de_input = torch.where(de_input!=1, de_input, pad)

            # record eos
            eos = torch.logical_or(eos, torch.eq(de_input, pad)) if i!=0 else torch.eq(de_input, pad)

            if(torch.equal(eos, torch.tensor([True for i in range(batch_size)], device=self.device).view(-1,1))):
                break

        return predict, distribution, hid_mean, hid_logvar, cell_mean, cell_logvar

    def inference(self, input_word, input_tense, target_tense, max_length):

        # get batch_size and length
        batch_size, length = input_word.shape

        # init encoder hidden and cell state (1 * batch_size * (hidden_size - condition_size) )
        en_hidden = self.encoder.inithidden(batch_size, self.hidden_size - self.condition_size)
        en_cell_state = self.encoder.initcell(batch_size, self.hidden_size - self.condition_size)

        # tense embedding (batch_size * 1 * condition_size) -> (1 * batch_size  * condition_size)
        input_tense_embedding = self.te_em(input_tense.view(-1, 1))
        input_tense_embedding = input_tense_embedding.permute(1, 0, 2)

        # cat condition into hidden and cell state
        en_hidden = torch.cat((en_hidden, input_tense_embedding), dim=2)
        en_cell_state = torch.cat((en_cell_state, input_tense_embedding), dim=2)
        # en_cell_state = torch.zeros_like(en_hidden, device=self.device)

        # encoder lstm
        for i in range(length):
            _, en_hidden, en_cell_state = self.encoder(input_word[:,i].view(-1,1), en_hidden, en_cell_state)

        # fully connection extract mean and logvar (1 * batch_size * latent_size)
        hid_mean = self.hid2mean(en_hidden)
        hid_logvar = self.hid2logvar(en_hidden)
        cell_mean = self.cell2mean(en_cell_state)
        cell_logvar = self.cell2logvar(en_cell_state)

        # reparameterize (1 * batch_size * latent_size)
        hid_latent = self.reparameterize(hid_mean, hid_logvar)
        cell_latent = self.reparameterize(cell_mean, cell_logvar)

        # decoder hidden and cell state (1 * batch_size * hidden_size)
        de_hidden = self.lat2hid(hid_latent)
        de_cell_state = self.lat2cell(cell_latent)

        target_tense_embedding = self.te_em(target_tense.view(-1, 1))
        target_tense_embedding = target_tense_embedding.permute(1, 0, 2)
        de_hidden = torch.cat((de_hidden, target_tense_embedding), dim=2)
        de_cell_state = torch.cat((de_cell_state, target_tense_embedding), dim=2)
        # de_cell_state = torch.zeros_like(de_hidden, device=self.device)
        
        # decoder input(SOS = 0)
        de_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        # PAD tensor
        pad = torch.tensor([2 for i in range(batch_size)], device=self.device).view(-1,1)

        # decoder lstm
        for i in range(max_length):
            output, de_hidden, de_cell_state = self.decoder(de_input, de_hidden, de_cell_state)

            # reconstruction
            number = torch.max(output, dim=2)[1]
            predict = torch.cat((predict, number), dim=1) if i!=0 else number

            de_input = number

            # transform eos to pad
            de_input = torch.where(de_input!=1, de_input, pad)

            # record eos
            eos = torch.logical_or(eos, torch.eq(de_input, pad)) if i!=0 else torch.eq(de_input, pad)

            if(torch.equal(eos, torch.tensor([True for i in range(batch_size)], device=self.device).view(-1,1))):
                break

        return predict

    def generate(self, tense, latent, cell, max_length):

        # tense shape -> 4 (sp, tp, pg, p)
        # latent shape -> 1 * 4 * latent size
        batch_size = latent.shape[1]

        # tense embedding (4 * 1 * condition_size) -> (1 * 4  * condition_size)
        tense_embedding = self.te_em(tense.view(-1, 1))
        tense_embedding = tense_embedding.permute(1, 0, 2)

        # decoder hidden and cell state (1 * 4 * hidden_size)
        de_hidden = self.lat2hid(latent)
        de_cell_state = self.lat2cell(cell)
        de_hidden = torch.cat((de_hidden, tense_embedding), dim=2)
        de_cell_state = torch.cat((de_cell_state, tense_embedding), dim=2)
        # de_cell_state = torch.zeros_like(de_hidden, device=self.device)

        # decoder input(SOS = 0)
        de_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        # PAD tensor
        pad = torch.tensor([2 for i in range(batch_size)], device=self.device).view(-1,1)

        # decoder lstm
        for i in range(max_length):
            output, de_hidden, de_cell_state = self.decoder(de_input, de_hidden, de_cell_state)

            # reconstruction
            number = torch.max(output, dim=2)[1]
            predict = torch.cat((predict, number), dim=1) if i!=0 else number

            de_input = number

            # transform eos to pad
            de_input = torch.where(de_input!=1, de_input, pad)

            # record eos
            eos = torch.logical_or(eos, torch.eq(de_input, pad)) if i!=0 else torch.eq(de_input, pad)

            if(torch.equal(eos, torch.tensor([True for i in range(batch_size)], device=self.device).view(-1,1))):
                break

        return predict

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, input, hidden, cell_state):
        embedded = self.embedding(input)
        output, (hidden, cell_state) = self.lstm(embedded, (hidden, cell_state))
        return output, hidden, cell_state

    def inithidden(self, batch_size, size):
        return torch.zeros(1, batch_size, size, device=self.device)

    def initcell(self, batch_size, size):
        return torch.zeros(1, batch_size, size, device=self.device)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, use_cuda):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, input, hidden, cell_state):
        output = self.embedding(input)
        output = F.relu(output)
        output, (hidden, cell_state) = self.lstm(output, (hidden, cell_state))
        output = self.softmax(self.out(output))
        return output, hidden, cell_state

    def inithidden(self, batch_size, size):
        return torch.zeros(1, batch_size, size, device=self.device)

    def initcell(self, batch_size, size):
        return torch.zeros(1, batch_size, size, device=self.device)
        
