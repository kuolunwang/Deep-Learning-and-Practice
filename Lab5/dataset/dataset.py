#!/usr/bin/env python3
from torch.utils.data.dataset import Dataset
from .converter import Converter
import torch
from torch.nn.utils.rnn import pad_sequence

class WordDataset(Dataset):
    def __init__(self, mode):
        self.converter = Converter()
        self.__readdata(mode)
        self.max_length = self.get_max_length()
        self.mode = mode

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        
        if(self.mode == "train"):
            return self.converter.word2tensor(self.words[index]), self.tenses[index]
        else:
            return self.converter.word2tensor(self.words[index][0]), \
                   torch.tensor(self.tenses[index][0]), \
                   self.converter.word2tensor(self.words[index][1]), \
                   torch.tensor(self.tenses[index][1])

    def __readdata(self, mode):

        path = "./dataset/train.txt" if mode == "train" else "./dataset/test.txt"

        self.words = []
        self.tenses = []

        with open(path, "r") as file:

            if(mode == "train"):
                for line in file:
                    self.words.extend(line.strip('\n').split(' '))
                    self.tenses.extend([i for i in range(4)])
            else:
                for line in file:
                    self.words.append(line.strip('\n').split(' '))
                    
                tenses = [["sp","p"], ["sp","pg"], ["sp","tp"], ["sp","tp"], ["p","tp"], ["sp","pg"], ["p","sp"], ["pg","sp"], ["pg","p"], ["pg","tp"]]
                for te in tenses:
                    self.tenses.append([self.converter.te2nb[t] for t in te])

    def get_max_length(self):

        max_length = 0
        for word in self.words:
            max_length = max(max_length, len(word))
        return max_length

    def collate_fn(self, batch):

        pad_data = pad_sequence([word for word, _ in batch], batch_first=True, padding_value=2) 
        # print(batch)
        _, tense = zip(*batch)

        return pad_data, torch.tensor(tense)
