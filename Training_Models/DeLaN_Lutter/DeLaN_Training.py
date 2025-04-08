'''
Autor:      Ole Uphaus
Datum:      08.04.2025
Beschreibung:
Dieses Skript soll das DeLaN neuronale Netz von Lutter Nutzen, um mit den Trainingsdaten des 2 FHG Roboters zu Trainieren. Es soll unterucht werden, wie gut das Modell performt, gegenüber dem Feed-Forward-NN.
'''

import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import onnx
import numpy as np

from DeLaN_model_Lutter import DeepLagrangianNetwork

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

    return features_training, labels_training, features_test, labels_test

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
        'learning_rate': 1.e-04,
        'weight_decay': 1.e-5,
        'max_epoch': 10000}

# Trainings- und Testdaten laden
features_training, labels_training, features_test, labels_test = extract_training_data('SimData__2025_04_04_09_51_52.mat')

features_training = torch.from_numpy(features_training[:2500, :]).float()
labels_training = torch.from_numpy(labels_training[:2500, :]).float()
features_test = torch.from_numpy(features_test[:2500, :]).float()
labels_test = torch.from_numpy(labels_test[:2500, :]).float()

# Dataset und Dataloader erstellen
dataset_training = TensorDataset(features_training, labels_training)
dataloader_training = DataLoader(dataset_training, batch_size=hyper["n_minibatch"], shuffle=True, drop_last=True, )
dataset_test = TensorDataset(features_test, labels_test)
dataloader_test = DataLoader(dataset_test, batch_size=hyper["n_minibatch"], shuffle=False, drop_last=False, )

# Modell Initialisieren
delan_model = DeepLagrangianNetwork(n_dof, **hyper)

# Generate & Initialize the Optimizer:
optimizer = torch.optim.Adam(delan_model.parameters(),
                                lr=hyper["learning_rate"],
                                weight_decay=hyper["weight_decay"],
                                amsgrad=True)

# Optimierung (Lernprozess)
num_epochs = hyper['max_epoch']  # Anzahl der Durchläufe durch den gesamten Datensatz

print('Starte Optimierung...')

for epoch in range(num_epochs):
    # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
    delan_model.train()
    loss_sum = 0

    for batch_features, batch_labels in dataloader_training:
        # Reset gradients:
        optimizer.zero_grad()

        # Variablen extrahieren
        q = batch_features[:, [0, 1]]
        qd = batch_features[:, [2, 3]]
        qdd = batch_features[:, [4, 5]]

        tau = batch_labels

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
    
    # Mittleren Loss berechnen und ausgeben
    training_loss_mean = loss_sum/len(dataloader_training)
   
    if epoch == 0 or np.mod(epoch + 1, 100) == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training-Loss: {training_loss_mean:.3e}')

# # Modellvalidierung mit Testdaten
# delan_model.eval()
# loss_sum = 0
# with torch.no_grad():
#     for batch_features, batch_labels in dataloader_test:
#         # Forward Pass
#         outputs = delan_model(batch_features).squeeze()
#         loss = criterion(outputs, batch_labels)

#         # Loss des aktuellen Batches ausfsummieren
#         loss_sum += loss.item()

# # Mittleren Loss berechnen und ausgeben
# test_loss_mean = loss_sum/len(dataloader_test)

# print(f'Anwenden des trainierten Modells auf unbekannte Daten, Test-Loss: {test_loss_mean:.6f}')

# if hyper_param['save_model'] == True:
#     # Dummy Input für Export (gleiche Form wie deine Eingabedaten) - muss gemacht werden
#     dummy_input = torch.randn(1, input_size)

#     # Aktueller Zeitstempel
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = os.path.join("Feedforward_NN", "Saved_Models", f"{timestamp}_feedforward_model.onnx")
#     scaler_path = os.path.join("Feedforward_NN", "Saved_Models", f"{timestamp}_scaler.mat")

#     # Modell exportieren
#     torch.onnx.export(model, dummy_input, model_path, 
#                     input_names=['input'], output_names=['output'], 
#                     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
#                     opset_version=14)

#     # Mittelwert und Std speichern
#     scipy.io.savemat(scaler_path, {
#         'mean_f': scaler_f.mean_,
#         'scale_f': scaler_f.scale_,
#         'mean_l': scaler_l.mean_,
#         'scale_l': scaler_l.scale_
#     })
