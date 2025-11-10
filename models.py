# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class trajectory2seq_elman(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_elman, self).__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.maxlen = maxlen

        # Definition des couches
        # Couches pour rnn
        # À compléter
        self.encoder_layer = nn.RNN(2,hidden_size=self.hidden_dim,num_layers=self.n_layers,batch_first=True)

        self.embedding = nn.Embedding(dict_size,hidden_dim)
        self.decoder_layer = nn.RNN(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.dict_size)
        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        # À compléter


    def encoder(self, x):

        hidden = None
        out, hidden = self.encoder_layer.forward(x,hidden)

        return out, hidden


    def decoder(self,encoders_out, hidden):


        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()

        vec_out = []
        for i in range(self.maxlen):
            embedded = self.embedding(vec_in)

            out, hidden = self.decoder_layer.forward(embedded,hidden)

            logits = self.fc.forward(out)

            vec_in = logits.argmax(dim=2)

            vec_out.append(vec_in)

        return vec_out, hidden

    def forward(self, x):

        out , h = self.encoder(x)
        out_vec, h = self.decoder(out,hidden=h)

        return out_vec
    

