'''
Autor:      Ole Uphaus
Datum:      08.04.2025
Beschreibung:
Dieses Skript ist eine Kopie vom DeLaN Trainingsskript und wird dafür verwendet, den optimalen Seed herauszufinden.
'''

import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
import numpy as np
import dill as pickle
import onnx
from pathlib import Path
import matplotlib.pyplot as plt

from DeLaN_model_Lutter import DeepLagrangianNetwork
from replay_memory_Lutter import PyTorchReplayMemory

def load_dataset(n_characters=3, filename="D:\\Programmierung_Ole\\Studienarbeit_Deep_Lagrangian_Networks\\Training_Models\\DeLaN_Lutter\\character_data.pickle", test_label=("e", "q", "v")):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    n_dof = 2

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    # test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean

def extract_training_data(file_name):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', 'MATLAB_Simulation', file_name)

    # Pfad normieren
    data_path = os.path.normpath(data_path)

    # Daten extrahieren
    data = scipy.io.loadmat(data_path)

    features_training = data['features_training']
    labels_training = data['labels_training']
    features_test = data['features_test']
    labels_test = data['labels_test']
    Mass_Cor_test = data['Mass_Cor_test']

    return features_training, labels_training, features_test, labels_test, Mass_Cor_test

# Seeds ausprobieren
max_seed = 100
seed_vec = np.arange(1, max_seed + 1)

# Ergebnisvektor
seed_loss_vec = np.zeros([max_seed, 3])
seed_loss_vec[:, 0] = seed_vec

# Loop, um mehrere Seeds auszuprobieren
for i_seed in seed_vec:

    print('Aktueller Seed: ', i_seed)

    # Set the seed:
    seed = i_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Parameter festlegen
    n_dof = 2
    hyper = {'n_width': 64,
            'n_depth': 2,
            'diagonal_epsilon': 0.01,
            'activation': 'SoftPlus',
            'b_init': 1.e-4,
            'b_diag_init': 0.001,
            'w_init': 'xavier_normal',
            'gain_hidden': np.sqrt(2.),
            'gain_output': 0.1,
            'n_minibatch': 512,
            'learning_rate': 5.e-04,
            'weight_decay': 1.e-5,
            'max_epoch': 2000,
            'save_model': False}

    # Checken, ob Cuda verfügbar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benutze Gerät: {device}")

    # Trainings- und Testdaten laden 
    features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_06_01_12_00_30_Samples_4096.mat')  # Mein Modell Trainingsdaten
    _, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_06_01_12_00_30_Samples_4096.mat')  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

    input_size = features_training.shape[1]

    # Trainingsdaten
    train1_qp = np.array(features_training[:, (0, 1)])
    train1_qv = np.array(features_training[:, (2, 3)])
    train1_qa = np.array(labels_training)
    train1_tau = np.array(features_training[:, (4, 5)])

    # Testdaten
    test1_qp = np.array(features_test[:, (0, 1)])
    test1_qv = np.array(features_test[:, (2, 3)])
    test1_qa = np.array(labels_test)
    test1_tau = np.array(features_test[:, (4, 5)])
    Mass_Cor_test = np.array(Mass_Cor_test)

    train_data, test_data, divider, dt_mean = load_dataset()    # Buchstaben Modell (Lutter)
    train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau = train_data
    test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data

    # Modell Initialisieren
    delan_model = DeepLagrangianNetwork(n_dof, **hyper).to(device)

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                    lr=hyper["learning_rate"],
                                    weight_decay=hyper["weight_decay"],
                                    amsgrad=True)

    # Generate Replay Memory:
    if device == torch.device('cuda'):
        cuda = True
    else:
        cuda = False

    mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
    mem = PyTorchReplayMemory(train1_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
    mem.add_samples([train1_qp, train1_qv, train1_qa, train1_tau])

    # Optimierung (Lernprozess)
    num_epochs = hyper['max_epoch']  # Anzahl der Durchläufe durch den gesamten Datensatz

    print('Starte Optimierung...')

    for epoch in range(num_epochs):
        # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
        delan_model.train()
        loss_sum = 0
        n_batches = 0

        for q, qd, qdd, tau in mem:
            # Reset gradients:
            optimizer.zero_grad()

            # Forward pass
            tau_hat, dEdt_hat = delan_model(q, qd, qdd)

            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)

            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, 2, 1).transpose(dim0=1, dim1=2), tau.view(-1, 2, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)

            # Compute gradients & update the weights:
            loss = l_mean_inv_dyn + l_mean_dEdt
            loss.backward()
            optimizer.step()

            # Loss des aktuellen Batches ausfsummieren
            loss_sum += loss.item()
            n_batches += 1
        
        # Mittleren Loss berechnen und ausgeben
        training_loss_mean = loss_sum/n_batches
    
        if epoch == 0 or np.mod(epoch + 1, 100) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training-Loss: {training_loss_mean:.3e}')

    # Modell evaluieren

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test1_qp).float().to(device)
    qd = torch.from_numpy(test1_qv).float().to(device)
    qdd = torch.from_numpy(test1_qa).float().to(device)

    # Prädiktion
    with torch.no_grad():

        # Loss berechnen
        tau_hat, dEdt_hat = delan_model(q, qd, qdd)
        tau_hat = tau_hat.cpu().numpy()
        dEdt_hat = dEdt_hat.cpu().numpy()

        err_inv = np.sum((tau_hat - test1_tau) ** 2, axis=1)
        l_mean_inv_dyn = np.mean(err_inv)

        dEdt = np.sum(qd.cpu().numpy() * test1_tau, axis=1)   # Numpy aquivalent zur Zeile oben aus Torch 
        err_dEdt = (dEdt_hat - dEdt) ** 2
        l_mean_dEdt = np.mean(err_dEdt)

        loss = l_mean_inv_dyn + l_mean_dEdt

    # Fehler abspeichern
    loss = l_mean_inv_dyn + l_mean_dEdt
    seed_loss_vec[i_seed - 1, 1] = loss
    seed_loss_vec[i_seed - 1, 2] = training_loss_mean

print(seed_loss_vec)

# Fehler Visualisieren
plt.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 1], label='Test-Loss')
plt.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 2], label='Training-Loss')
plt.xscale("linear")
plt.yscale("log") 
plt.xlabel("Seed")
plt.ylabel("Fehler MSE")
plt.title("Zusammenhang zwischen Seed und Fehler")
plt.legend()
plt.grid(True)
plt.show()