import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import pickle

class HandwrittenWords(Dataset):
    """Ensemble de donnees de mots ecrits a la main."""

    def __init__(self, filename):
        # Lecture du text
        self.pad_symbol     = pad_symbol = '<pad>'
        self.start_symbol   = start_symbol = '<sos>'
        self.stop_symbol    = stop_symbol = '<eos>'


        self.data = dict()
        with open(filename, 'rb') as fp:
            self.data = pickle.load(fp)

        maxT = max(len(seq[1][0]) for seq in self.data)
        maxLabel = max(len(seq[0]) for seq in self.data)

        wantedLenT = maxT + 1

        wantedLenT = 458

        wantedLenLabel = maxLabel + 1

        wantedLenLabel = 6

        self.maxlen = wantedLenLabel

        self.symb2int = dict()
        self.symb2int = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=3):
            self.symb2int[c] = i
        cpt_symb_fr = 3

        minX = 1000
        minY = 1000
        maxX = -1000
        maxY = -1000

        for i, dataentry in enumerate(self.data):

            inputseq = dataentry[1]
            label = dataentry[0]
            label = list(label)

            liveminX = min(inputseq[0])
            liveminY = min(inputseq[1])
            livemaxX = max(inputseq[0])
            livemaxY = max(inputseq[1])

            if liveminX < minX:
                minX = liveminX
            if liveminY < minY:
                minY = liveminY
            if livemaxX > maxX:
                maxX = livemaxX
            if livemaxY > maxY:
                maxY = livemaxY


            for sym in label:
                if sym not in self.symb2int:
                    self.symb2int[sym] = cpt_symb_fr
                    cpt_symb_fr += 1

            label.append(self.stop_symbol)
            label = np.array(label)

            lastx = inputseq[0][-1]
            lasty = inputseq[1][-1]

            N = wantedLenT - inputseq.shape[1]
            paddingArray = np.tile(np.array([[lastx], [lasty]]), (1, N))
            inputseq = np.append(inputseq,paddingArray,axis=1)

            paddingArray = np.full([wantedLenLabel - label.shape[0]],self.pad_symbol)

            label = np.append(label,paddingArray)


            self.data[i] = (label,inputseq)

        maxX = 2976.344897715093
        maxY = 678.7003612207204
        minX = -226.13771862085716
        minY = -588.6337252912118

        for i, dataentry in enumerate(self.data):
            inputseq = dataentry[1]
            x = inputseq[0,:]
            y = inputseq[1,:]

            x_norm = (x - minX) / (maxX - minX)
            y_norm = (y - minY) / (maxY - minY)

            norm_inputseq = np.array([x_norm,y_norm])

            self.data[i] = (self.data[i][0], norm_inputseq)


        self.int2symb = dict()
        self.int2symb =  {v:k for k,v in self.symb2int.items()}

        self.dict_size = len(self.symb2int)



    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        # À compléter


        target = self.data[idx][0]
        traget_int = [self.symb2int[i] for i in target]

        return torch.tensor(self.data[idx][1], dtype=torch.float32) ,torch.tensor(traget_int)

    def visualisation(self, idx):

        pass
        

if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords('data_trainval.p')
    call = a[0]

    pass
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))