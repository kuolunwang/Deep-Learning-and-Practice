#!/usr/bin/env python3

import torch

class Converter():
    def __init__(self):

        self.te2nb = {"sp":0, "tp":1, "pg":2, "p":3}
        self.nb2te = {0:"sp", 1:"tp", 2:"pg", 3:"p"}
        self.__setup_al2nb()
        self.__setup_nb2al()

    def __setup_al2nb(self):
        self.al2nb = {chr(i+97):i+3 for i in range(26)}
        self.al2nb.update({"SOS":0, "EOS":1, "PAD":2})

    def __setup_nb2al(self):
        self.nb2al = {i+3:chr(i+97) for i in range(26)}
        self.nb2al.update({0:"SOS", 1:"EOS", 2:"PAD"})

    def word2tensor(self, word):
        ten = []
        for i in word:
            ten.append(self.al2nb[i])
        ten.append(self.al2nb["EOS"])
        return torch.tensor(ten)

    def tensor2word(self, tensor):
        word  = ""
        for i in range(tensor.shape[0]):
            word += self.nb2al[tensor[i].item()] if tensor[i].item() >= 3 else ""
        return word
