import pandas as pd
import torch
import re
from torch.utils.data import Dataset

class Tokenizer:
    def __init__(self):
        self.wrd = []
        self.wrd2idx = {}
        self.idx2wrd = {}
        self.last = -1
    def addWord(self, word):
        if(word in self.wrd):
            print("The word is already in the dictionary. Stupid monkey >:(")
            return 
        self.wrd2idx[word] = self.last
        self.idx2wrd[self.last] = word
        self.last += 1
        self.wrd.append(word)
    def Word2Idx(self, word):
        if(not word in self.wrd):
            self.addWord(word)
        return self.wrd2idx[word]
    def Idx2Word(self, idx):
        if(idx >= self.last):
            print("Index out of range. Stupid monkey >:(")
        else:
            return self.idx2wrd[idx]
class SpamData(Dataset):
    def __init__(self, path, dict, max_size):
        self.file = pd.read_csv(path)
        self.dict = dict
        self.pattern = r'[^a-zA-Z0-9\s]'
        self.max_size = max_size
    def __len__(self):
        return len(self.file)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.file.iloc[idx, 0]
        label = self.dict.Word2Idx(label)
        text = [-1] * self.max_size
        text_tmp = self.file.iloc[idx, 1]
        text_tmp = re.sub(self.pattern, '', text_tmp)
        text_tmp = text_tmp.split()
        for i in range(len(text_tmp)):
            text[i] = self.dict.Word2Idx(text_tmp[i])
        text = torch.tensor(text, dtype=torch.float)
        sample = (text, label)
        return sample
    def get(self):
        return self.dict
