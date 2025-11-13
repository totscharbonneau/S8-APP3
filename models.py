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
        vec_out = torch.zeros((batch_size, self.maxlen, self.dict_size)).to(self.device)

        for i in range(self.maxlen):
            embedded = self.embedding(vec_in)

            out, hidden = self.decoder_layer.forward(embedded,hidden)

            logits = self.fc.forward(out)

            vec_in = logits.argmax(dim=2)

            vec_out[:, i, :] = logits.squeeze(1)

        return vec_out, hidden

    def forward(self, x):

        out , h = self.encoder(x)
        out_vec, h = self.decoder(out,hidden=h)

        return out_vec, h
    





class trajectory2seq_gru_att_bi(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_gru_att_bi, self).__init__()
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

        self.embedding_encode = nn.Linear(2, hidden_dim)

        self.encoder_layer = nn.GRU(self.hidden_dim,self.hidden_dim,self.n_layers,batch_first=True,bidirectional=True)

        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.embedding_decode = nn.Embedding(self.dict_size,hidden_dim)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)



        # Couches pour attention
        self.softmax = nn.Softmax(dim=-1)
        self.att_combine = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

        # Couche dense pour la sortie
        # À compléter
        self.fc = nn.Linear(hidden_dim, self.dict_size)


    def attentionModule(self,query,values):

        query = self.query_projection.forward(query)

        values_switch = values.reshape(-1,values.shape[2],values.shape[1])
        step1 = torch.bmm(query,values_switch)
        W = self.softmax.forward(step1)
        attention_output = torch.bmm(W,values)

        return attention_output , W



    def encoder(self, x):

        hidden = None
        x_encoded = self.embedding_encode(x)
        out, hidden = self.encoder_layer.forward(x_encoded,hidden)

        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)

        hidden = self.hidden_projection(hidden)

        return out, hidden


    def decoder(self,encoders_out, hidden):


        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, self.maxlen, self.dict_size)).to(self.device)

        for i in range(self.maxlen):
            embedded = self.embedding_decode(vec_in)

            out, hidden = self.decoder_layer.forward(embedded,hidden)

            attention_out, attention_weights = self.attentionModule(out, encoders_out)


            attention_and_decode = torch.concat([attention_out,out],dim=2)

            a_d_compact = self.att_combine.forward(attention_and_decode)

            logits = self.fc.forward(a_d_compact)

            vec_in = logits.argmax(dim=2)

            vec_out[:, i, :] = logits.squeeze(1)

        return vec_out, hidden

    def forward(self, x):

        out , h = self.encoder(x)
        out_vec, h = self.decoder(out,hidden=h)

        return out_vec, h



class trajectory2seq_gru_att(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_gru_att, self).__init__()
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

        self.embedding_encode = nn.Linear(2, hidden_dim)

        self.encoder_layer = nn.GRU(self.hidden_dim,self.hidden_dim,self.n_layers,batch_first=True)

        # self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.embedding_decode = nn.Embedding(self.dict_size,hidden_dim)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)



        # Couches pour attention
        self.softmax = nn.Softmax(dim=-1)
        self.att_combine = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Couche dense pour la sortie
        # À compléter
        self.fc = nn.Linear(hidden_dim, self.dict_size)


    def attentionModule(self,query,values):

        query = self.query_projection.forward(query)

        values_switch = values.reshape(-1,values.shape[2],values.shape[1])
        step1 = torch.bmm(query,values_switch)
        W = self.softmax.forward(step1)
        attention_output = torch.bmm(W,values)

        return attention_output , W



    def encoder(self, x):

        hidden = None
        x_encoded = self.embedding_encode(x)
        out, hidden = self.encoder_layer.forward(x_encoded,hidden)

        return out, hidden


    def decoder(self,encoders_out, hidden):


        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, self.maxlen, self.dict_size)).to(self.device)

        for i in range(self.maxlen):
            embedded = self.embedding_decode(vec_in)

            out, hidden = self.decoder_layer.forward(embedded,hidden)

            attention_out, attention_weights = self.attentionModule(out, encoders_out)


            attention_and_decode = torch.concat([attention_out,out],dim=2)

            a_d_compact = self.att_combine.forward(attention_and_decode)

            logits = self.fc.forward(a_d_compact)

            vec_in = logits.argmax(dim=2)

            vec_out[:, i, :] = logits.squeeze(1)

        return vec_out, hidden

    def forward(self, x):

        out , h = self.encoder(x)
        out_vec, h = self.decoder(out,hidden=h)

        return out_vec, h


class trajectory2seq_gru_bi(nn.Module):
    def __init__(self, hidden_dim, n_layers, int2symb, symb2int, dict_size, device, maxlen):
        super(trajectory2seq_gru_bi, self).__init__()
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
        self.embedding_encode = nn.Linear(2, hidden_dim)

        self.encoder_layer = nn.GRU(self.hidden_dim,self.hidden_dim,self.n_layers,batch_first=True,bidirectional=True)

        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.embedding_decode = nn.Embedding(self.dict_size,hidden_dim)
        self.decoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, self.dict_size)
        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        # À compléter


    def encoder(self, x):

        hidden = None
        x_encoded = self.embedding_encode(x)
        out, hidden = self.encoder_layer.forward(x_encoded,hidden)

        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        hidden = self.hidden_projection(hidden)
        return out, hidden


    def decoder(self,encoders_out, hidden):


        batch_size = hidden.shape[1]
        vec_in = torch.zeros((batch_size, 1)).to(self.device).long()
        vec_out = torch.zeros((batch_size, self.maxlen, self.dict_size)).to(self.device)

        for i in range(self.maxlen):
            embedded = self.embedding_decode(vec_in)

            out, hidden = self.decoder_layer.forward(embedded,hidden)

            logits = self.fc.forward(out)

            vec_in = logits.argmax(dim=2)

            vec_out[:, i, :] = logits.squeeze(1)

        return vec_out, hidden

    def forward(self, x):

        out , h = self.encoder(x)
        out_vec, h = self.decoder(out,hidden=h)

        return out_vec, h