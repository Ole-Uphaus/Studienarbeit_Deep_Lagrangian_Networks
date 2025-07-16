'''
Autor:      Ole Uphaus
Datum:      20.03.2025
Beschreibung:
Dieses Skript soll auf Grundlage der Simuleirten Trainingsdaten ein neuronales Netz trainieren. Dies ist jedoch nur ein Feed-Forward Netz, das keine Informationen über lagrange Gleichungen enthält. Die benötigten Trainingsdaten werden aus einem .mat File extrahiert.
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
from pathlib import Path
import numpy as np
import sys

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
DeLaN_dir_path = os.path.join(script_path, '..', 'DeLaN_Ole')

if DeLaN_dir_path not in sys.path:
    sys.path.insert(0, DeLaN_dir_path)

from FFNN_model import FeedForwardNN
from DeLaN_functions_Ole import *

# Checken, ob Cuda verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Gerät: {device}")

# Seed setzten für Reprodizierbarkeit
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Parameter festlegen
hyper_param = {
    # Netzparameter
    'hidden_size': 256,
    'depth': 2,

    # Training
    'epoch': 100,
    'learning_rate': 0.001,
    'wheight_decay': 1e-5,
    'dropout': 0.3,
    'batch_size': 512,

    # Sonstiges
    'save_model': False,
    }

# Trainings- und Testdaten laden
target_folder = 'MATLAB_Simulation' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen'
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Daten vorbereiten (scaling)
scaler_f = StandardScaler()
scaler_l = StandardScaler()

scaled_features_training = scaler_f.fit_transform(features_training)
scaled_labels_training = scaler_l.fit_transform(labels_training)
scaled_features_test = scaler_f.transform(features_test)    # Hier nur transform, um Skalierungsparameter beizubehalten
scaled_labels_test = scaler_l.transform(labels_test)    # Hier nur transform, um Skalierungsparameter beizubehalten

# Trainings- und Testdaten in Torch-Tensoren umwandeln
features_tensor_training = torch.tensor(scaled_features_training, dtype=torch.float32)
labels_tensor_training = torch.tensor(scaled_labels_training, dtype=torch.float32)
features_tensor_test = torch.tensor(scaled_features_test, dtype=torch.float32).to(device)
labels_tensor_test = torch.tensor(scaled_labels_test, dtype=torch.float32).to(device)

# Neuronales Netz initialisieren
input_size = labels_training.shape[1]
n_dof = labels_training.shape[1]
model = FeedForwardNN(input_size, n_dof, **hyper_param).to(device)

# Loss funktionen und Optimierer wählen
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=hyper_param['learning_rate'], weight_decay=hyper_param['wheight_decay'])

# Dataset und Dataloader erstellen
dataset_training = TensorDataset(features_tensor_training, labels_tensor_training)
dataloader_training = DataLoader(dataset_training, batch_size=hyper_param['batch_size'], shuffle=True, drop_last=True, )

# Optimierung (Lernprozess)
num_epochs = hyper_param['epoch']  # Anzahl der Durchläufe durch den gesamten Datensatz

print('Starte Optimierung...')

for epoch in range(num_epochs):
    # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
    model.train()
    loss_sum = 0

    for batch_features, batch_labels in dataloader_training:

        # Tensoren auf GPU schieben
        batch_features = batch_features.to(device)  # q, qp, qpp
        tau = batch_labels.to(device)  # tau

        # Forward pass
        tau_hat = model(batch_features).squeeze()
        
        # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
        err_inv_dyn = torch.sum((tau_hat - tau)**2, dim=1)
        mean_err_inv_dyn = torch.mean(err_inv_dyn)

        loss = mean_err_inv_dyn

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss des aktuellen Batches ausfsummieren
        loss_sum += loss.item()
    
    # Mittleren Loss berechnen und ausgeben
    training_loss_mean = loss_sum/len(dataloader_training)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training-Loss: {training_loss_mean:.6f}')


