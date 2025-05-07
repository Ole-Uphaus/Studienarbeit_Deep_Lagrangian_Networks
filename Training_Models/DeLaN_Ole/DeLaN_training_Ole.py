'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
In diesem Skript wird das von mir erstellte Deep Lagrangien Network trainiert. Hier ist Raum für allgemeine Tests und Erprobungen der Hyperparameter.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

from DeLaN_model_Ole import Deep_Lagrangian_Network
from DeLaN_functions_Ole import *

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Seed setzen für Reproduzierbarkeit
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parameter festlegen
hyper_param = {
    'hidden_width': 32,
    'hidden_depth': 3,
    'L_diagonal_offset': 0.001,
    'activation_fnc': 'softplus',
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-5,
    'n_epoch': 200,
    'save_model': False}

# Trainings- und Testdaten laden 
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_01_08_35_27_Samples_3000.mat')  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_01_08_35_27_Samples_3000.mat')  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Torch Tensoren der Trainingsdaten erstellen
features_training_tensor = torch.tensor(features_training, dtype=torch.float32)
labels_training_tensor = torch.tensor(labels_training, dtype=torch.float32)

# Dataset und Dataloader für das Training erstellen
dataset_training = TensorDataset(features_training_tensor, labels_training_tensor)
dataloader_training = DataLoader(dataset_training, batch_size=hyper_param['batch_size'], shuffle=True)

# Ausgabe Datendimensionen
print('Datenpunkte Training: ', features_training.shape[0])
print('Datenpunkte Evaluierung: ', features_test.shape[0])
print()

# Testnetzwerk erstellen
n_dof = labels_training.shape[1]
DeLaN_network = Deep_Lagrangian_Network(n_dof, **hyper_param).to(device)

# Optimierer Initialisieren
optimizer = torch.optim.Adam(DeLaN_network.parameters(),
                                lr=hyper_param["learning_rate"],
                                weight_decay=hyper_param["weight_decay"])

# Optimierung starten und Zeitmessung beginnen
print('Starte Optimierung...')
print()
start_time = time.time()

# Training des Netzwerks
for epoch in range(hyper_param['n_epoch']):
    # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
    DeLaN_network.train()
    loss_sum = 0

    for batch_features, batch_labels in dataloader_training:
        # Gradienten zurücksetzen
        optimizer.zero_grad()

        # Trainingsdaten zuordnen
        q = batch_features[:, (0, 1)].to(device)
        qd = batch_features[:, (2, 3)].to(device)
        qdd = batch_labels.to(device)
        tau = batch_features[:, (4, 5)].to(device)

        # Forward pass
        tau_hat, _, _, _ = DeLaN_network(q, qd, qdd)

        # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
        err_inv_dyn = torch.sum((tau_hat - tau)**2, dim=1)
        mean_err_inv_dyn = torch.mean(err_inv_dyn)

        # Loss berechnen und Optimierungsschritt durchführen
        loss = mean_err_inv_dyn
        loss.backward()
        torch.nn.utils.clip_grad_norm_(DeLaN_network.parameters(), max_norm=0.5)
        optimizer.step()

        # Loss des aktuellen Batches aufsummieren
        loss_sum += loss.item()

    # Mittleren Loss berechnen und ausgeben
    loss_mean_batch = loss_sum/len(dataloader_training)

    if epoch == 0 or np.mod(epoch + 1, 100) == 0:
        print(f'Epoch [{epoch + 1}/{hyper_param['n_epoch']}], Training-Loss: {loss_mean_batch:.3e}, Verstrichene Zeit: {(time.time() - start_time):.2f} s')

