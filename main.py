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
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    lr = 0.01


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

    # Instanciation de l'ensemble de données
    # À compléter
    dataset = HandwrittenWords("data_trainval.p")

    n_train_samp = int(len(dataset)*train_val_split)
    n_val_samp = len(dataset)-n_train_samp
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])


    dataload_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)


    # Instanciation du model
    model = trajectory2seq_elman(hidden_dim=n_hidden,n_layers=n_layers,int2symb=dataset.int2symb, \
                                 symb2int=dataset.symb2int,dict_size=dataset.dict_size,device=device,maxlen=dataset.maxlen)


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

                # input_seq = input_seq.to(device).long()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad()

                output, hidden = model(input_seq)
                loss = criterion(output, target_seq)

                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

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