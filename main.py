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


class predictor_visuliser():
    def __init__(self, int2sym):
        self.int2sym = int2sym

    def __call__(self, input_seq, target_seq, prediction):
        input_seq = input_seq
        target_seq = target_seq.cpu().numpy()
        prediction = prediction[0]
        prediction = prediction.argmax(dim=1)
        prediction = prediction.cpu().numpy()
        target_string = [self.int2sym[i] for i in target_seq[0]]

        prediction_string = [self.int2sym[i] for i in prediction]

        target_string = "Target : " + "".join(target_string)
        prediction_string ="Prediction : " + "".join(prediction_string)

        x = input_seq[0, :].cpu().numpy()
        y = input_seq[1, :].cpu().numpy()

        mask = (x != 1e6) & (x != 2e6) & (y != 1e6) & (y != 2e6)
        x_filtered = x[mask]
        y_filtered = y[mask]

        plt.figure(figsize=(8, 6))
        plt.plot(x_filtered, y_filtered, marker='o', linestyle='-', markersize=3)  # line plot with points
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Trajectory over time')
        plt.grid(False)
        plt.subplots_adjust(bottom=0.22)


        plt.figtext(0.5, 0.12, target_string, ha="center", fontsize=10)
        plt.figtext(0.5, 0.05, prediction_string, ha="center", fontsize=10)
        plt.show()
        pass


if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = False           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = False     # Génération images test?
    seed = 88                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)
    lr = 0.005


    # À compléter
    n_epochs = 100
    train_val_split = 0.7
    batch_size = 50
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
    visualizator = predictor_visuliser(dataset.int2symb)

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
        val_dist = []
        val_loss = []
        fig, ax = plt.subplots(1)  # Initialisation figure

        criterion = nn.CrossEntropyLoss(ignore_index=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                input_seq, target_seq = data

                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad()

                test_size = input_seq.reshape((-1,458,2))

                output, hidden = model(test_size)


                # visualizator(input_seq,target_seq,output)


                loss = criterion(output.permute(0, 2, 1), target_seq)

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

            # ✅ VALIDATION
            model.eval()  # Set model to evaluation mode
            running_loss_val = 0
            dist_val = 0

            with torch.no_grad():  # Disable gradient computation
                for batch_idx, data in enumerate(dataload_val):
                    input_seq, target_seq = data

                    input_seq = input_seq.to(device)
                    target_seq = target_seq.to(device).long()

                    test_size = input_seq.reshape((-1, 458, 2))

                    output, hidden = model(test_size)

                    loss = criterion(output.permute(0, 2, 1), target_seq)
                    running_loss_val += loss.item()

                    output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                    target_seq_list = target_seq.cpu().tolist()

                    for i in range(len(output_list)):
                        a = target_seq_list[i]
                        b = output_list[i]
                        Ma = a.index(1) if 1 in a else len(a)
                        Mb = b.index(1) if 1 in b else len(b)
                        dist_val += edit_distance(a[:Ma], b[:Mb]) / len(output_list)

            avg_val_loss = running_loss_val / len(dataload_val)
            avg_val_dist = dist_val / len(dataload_val)

            # print('Val   - Epoch: {}/{} Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
            #     epoch, n_epochs, avg_val_loss, avg_val_dist))

            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                train_dist.append(dist / len(dataload_train))
                val_loss.append(avg_val_loss)
                val_dist.append(avg_val_dist)

                ax.cla()
                ax.plot(train_loss, label='training loss')
                ax.plot(train_dist, label='training distance')
                ax.plot(val_loss, label='validation loss')
                ax.plot(val_dist, label='validation distance')
                ax.legend()
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss / Distance')
                plt.draw()
                plt.pause(0.01)

            if epoch == 1 or avg_val_loss < min(val_loss[:-1] if len(val_loss) > 1 else [float('inf')]):
                torch.save(model.state_dict(), 'best_model.pth')
                # print('✓ Model saved!')

            # print()

    if test:

        viz = predictor_visuliser(dataset.int2symb)

        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        test_dataset = HandwrittenWords("data_test.p")
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        running_loss_test = 0
        dist_test = 0

        criterion = nn.CrossEntropyLoss(ignore_index=2)

        with torch.no_grad():
            for i in range(10):
                input_seq, target_seq = test_dataset[np.random.randint(0,len(test_dataset))]

                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device).long()
                target_seq = target_seq.unsqueeze(0)
                test_size = input_seq.reshape((-1, 458, 2))

                output, hidden = model(test_size)

                viz(input_seq,target_seq,output)

                loss = criterion(output.permute(0, 2, 1), target_seq)

                print(loss.item())
                running_loss_test += loss.item()

                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()

                for i in range(len(output_list)):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1) if 1 in a else len(a)
                    Mb = b.index(1) if 1 in b else len(b)
                    dist_test += edit_distance(a[:Ma], b[:Mb]) / len(output_list)

        avg_test_loss = running_loss_test / len(test_dataloader)
        avg_test_dist = dist_test / len(test_dataloader)

        print('Test Results: Loss: {:.6f}, Edit Distance: {:.6f}'.format(avg_test_loss, avg_test_dist))




