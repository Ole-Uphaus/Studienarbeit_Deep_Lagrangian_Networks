'''
Autor:      Ole Uphaus
Datum:      20.03.2025
Beschreibung:
Dieses Skript soll auf Grundlage der Simuleirten Trainingsdaten ein neuronales Netz trainieren. Dies ist jedoch nur ein Feed-Forward Netz, das keine Informationen über lagrange Gleichungen enthält. Die benötigten Trainingsdaten werden aus einem .mat File extrahiert.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
DeLaN_dir_path = os.path.join(script_path, '..', 'DeLaN_Ole')

if DeLaN_dir_path not in sys.path:
    sys.path.insert(0, DeLaN_dir_path)

from FFNN_model import FeedForwardNN
from DeLaN_functions_Ole import *

def model_evaluation(model, features_tensor_test, labels_tensor_test):
    # Forward pass
    tau_hat_eval = model(features_tensor_test).squeeze()

    # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
    err_inv_dyn = torch.sum((tau_hat_eval - labels_tensor_test)**2, dim=1)
    mean_err_inv_dyn = torch.mean(err_inv_dyn)

    return tau_hat_eval.cpu().detach().numpy(), mean_err_inv_dyn.cpu().detach().numpy()

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
    'activation_fnc': 'relu',

    # Training
    'epoch': 1000,
    'learning_rate': 0.001,
    'wheight_decay': 1e-4,
    'dropout': 0.3,
    'batch_size': 512,

    # Sonstiges
    'save_model': False,
    }

# Trainings- und Testdaten laden
target_folder = 'MATLAB_Simulation' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen'
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Trainings- und Testdaten in Torch-Tensoren umwandeln
features_tensor_training = torch.tensor(features_training, dtype=torch.float32)
labels_tensor_training = torch.tensor(labels_training, dtype=torch.float32)
features_tensor_test = torch.tensor(features_test, dtype=torch.float32).to(device)
labels_tensor_test = torch.tensor(labels_test, dtype=torch.float32).to(device)

# Neuronales Netz initialisieren
input_size = features_training.shape[1]
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
start_time = time.time()

# Training des Netzwerks
training_loss_history = []
test_loss_history = []

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)    # Gradienten Clipping für besseres Training
        optimizer.step()

        # Loss des aktuellen Batches ausfsummieren
        loss_sum += loss.item()
    
    # Mittleren Loss berechnen und ausgeben
    training_loss_mean = loss_sum/len(dataloader_training)

    # Evaluierung auf Testdaten
    _, test_loss = model_evaluation(model, features_tensor_test, labels_tensor_test)
    
    # Loss an History anhängen
    training_loss_history.append([epoch + 1, training_loss_mean])
    test_loss_history.append([epoch + 1, test_loss])

    if epoch == 0 or np.mod(epoch + 1, 100) == 0:

        print(f'Epoch [{epoch+1}/{num_epochs}], Training-Loss: {training_loss_mean:.3e}, Test-Loss: {test_loss:.3e}, Verstrichene Zeit: {(time.time() - start_time):.2f} s')

# Modell evaluieren
tau_hat_test, test_loss = model_evaluation(model, features_tensor_test, labels_tensor_test)

# Tau test (berechnen, da im Folgenden oft gebraucht)
tau_test = labels_tensor_test.cpu().detach().numpy()

# Metriken Berechnen (Fehler aus inverser Dynamik)
mse_tau = np.mean(np.sum((tau_hat_test - tau_test)**2, axis=1))
rmse_tau = np.sqrt(mse_tau)
tau_mean = np.sqrt(np.mean(np.sum((tau_test)**2, axis=1)))
rmse_tau_percent = rmse_tau/tau_mean*100

# Metriken ausgeben
print('Metriken:')
print(f"MSE Test: {mse_tau:4f}")
print(f"RMSE (Absolutfehler) Test: {rmse_tau:4f}")
print(f"Prozentualer Fehler Test: {rmse_tau_percent:4f}")

# Plotten
samples_vec = np.arange(1, tau_hat_test.shape[0] + 1)

# Loss Entwicklung plotten
training_loss_history = np.array(training_loss_history)
test_loss_history = np.array(test_loss_history)

plt.figure()

plt.semilogy(training_loss_history[:, 0], training_loss_history[:, 1], label='Training Loss')
plt.semilogy(test_loss_history[:, 0], test_loss_history[:, 1], label='Test Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.title('Loss-Verlauf während des Trainings')
plt.grid(True)
plt.legend()

# tau
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(samples_vec, tau_hat_test[:, 0], label='tau1 FFNN')
plt.plot(samples_vec, tau_test[:, 0] ,label='tau1 Analytic')
plt.title('tau1')
plt.xlabel('Samples')
plt.ylabel('tau')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(samples_vec, tau_hat_test[:, 1], label='tau2 FFNN')
plt.plot(samples_vec, tau_test[:, 1] ,label='tau2 Analytic')
plt.title('tau2')
plt.xlabel('Samples')
plt.ylabel('tau')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()