# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = False                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = False     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    lr = 0.001


    # À compléter
    n_epochs = 20
    train_val_split = 0.7
    batch_size = 100
    n_hidden = 20               # Nombre de neurones caches par couche
    n_layers = 2               # Nombre de de couches

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(device)

    # Instanciation de l'ensemble de données
    # À compléter
    dataset = HandwrittenWords("data_trainval.p")

    n_train_samp = int(len(dataset)*train_val_split)
    n_val_samp = len(dataset)-n_train_samp
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])


    dataload_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    print('Number of epochs : ', n_epochs)
    print('Training data : ', len(dataset))
    print('Taille dictionnaires: ', dataset.dict_size)
    print('\n')


    # Instanciation du model
    model = trajectory2seq_gru(hidden_dim=n_hidden,n_layers=n_layers,int2symb=dataset.int2symb, \
                                 symb2int=dataset.symb2int,dict_size=dataset.dict_size,device=device,maxlen=dataset.maxlen)

    model = model.to(device)
    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters() ]))

    # Initialisation des variables
    # À compléter

    if trainning:

        # Fonction de coût et optimizateur
        # À compléter
        train_dist = []  # Historique des distances
        train_loss = []  # Historique des coûts
        fig, ax = plt.subplots(1)  # Initialisation figure

        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0

            for batch_idx, data in enumerate(dataload_train):
                input_seq, target_seq = data

                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad()

                test_size = input_seq.reshape((-1,458,2))

                output, hidden = model(test_size)

                loss = criterion(output.reshape(-1,29,6), target_seq)

                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()

                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1) # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)# longueur mot b
                    dist += edit_distance(a[:Ma],b[:Mb])/batch_size


                print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * batch_size, len(dataload_train.dataset),
                    100. * batch_idx *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx+1) * batch_size, len(dataload_train.dataset),
                    100. * (batch_idx+1) *  batch_size / len(dataload_train.dataset), running_loss_train / (batch_idx + 1),
                    dist/len(dataload_train)), end='\r')

            if learning_curves:
                train_loss.append(running_loss_train/len(dataload_train))
                train_dist.append(dist/len(dataload_train))
                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.legend()
                plt.draw()
                plt.pause(0.01)


            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage


    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass