'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
In diesem Skript wird das von mir erstellte Deep Lagrangien Network trainiert. Hier ist Raum für allgemeine Tests und Erprobungen der Hyperparameter.
'''

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt

from DeLaN_training_Ole import *
from DeLaN_functions_Ole import *

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Parameter festlegen
hyper_param = {
    # Netzparameter
    'hidden_width': 64,
    'hidden_depth': 2,
    'activation_fnc': 'elu',
    'activation_fnc_diag': 'softplus',

    # Initialisierung
    'bias_init_constant': 1.e-3,
    'wheight_init': 'xavier_normal',

    # Lagrange Dynamik
    'L_diagonal_offset': 1.e-2,
    
    # Training
    'dropuot': 0.0,
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-4,
    'n_epoch': 2000,

    # Reibungsmodell
    'use_friction_model': False,
    'friction_model_init_d': [0.01, 0.01],
    'friction_model_init_c': [0.01, 0.01],
    'friction_model_init_s': [0.01, 0.01],
    'friction_model_init_v': [0.01, 0.01],
    'friction_epsilon': 100.0,

    # Sonstiges
    'use_inverse_model': True,
    'use_forward_model': True,
    'use_energy_consumption': False,
    'save_model': False}

# Trainings- und Testdaten laden
target_folder = 'Studienarbeit_Data' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen' 'Studienarbeit_Data'
features_training, labels_training, _, _, _ = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Seeds festlegen
max_seed = 100
seed_vec = np.arange(1, max_seed + 1)

# Ergebnisvektor
seed_loss_vec = np.zeros([max_seed, 3])
seed_loss_vec[:, 0] = seed_vec

# Loop, um mehrere Seeds auszuprobieren
for i_seed in seed_vec:

    print('Aktueller Seed: ', i_seed)

    # Seed setzen
    seed = i_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Modell trainieren
    DeLaN_network, results = Delan_Train_Eval(
            target_folder,
            features_training,
            labels_training,
            features_test,
            labels_test,
            hyper_param,
            device
        )
    
    # Ergebnisse entpacken
    training_loss_history = np.array(results['training_loss_history'])
    test_loss_history = np.array(results['test_loss_history'])

    # Kleinsten evaluierungsfehler finden
    error_eval = np.min(test_loss_history[:, 1])
    min_idx = np.argmin(test_loss_history[:, 1])
    error_train = training_loss_history[min_idx, 1]

    # Fehler abspeichern
    seed_loss_vec[i_seed - 1, 1] = error_train # Fehler Training
    seed_loss_vec[i_seed - 1, 2] = error_eval # Fehler Evaluierung

# Ausgabe Fehlervektor
print('Ergebnisvektor: \n', seed_loss_vec)

# Fehlervektor abspeichern für spätere Plots
script_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_path, '..', 'DeLaN_Plots', 'Ergebnisse_Seed_Untersuchung.npy')

np.save(save_path, seed_loss_vec)

# Fehler Visualisieren
plt.figure()
plt.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 1], label='Training-Loss')
plt.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 2], label='Test-Loss')
plt.xscale("linear")
plt.yscale("log") 
plt.xlabel("Seed")
plt.ylabel("Fehler MSE")
plt.title("Zusammenhang zwischen Seed und Fehler")
plt.grid(True)
plt.legend()
plt.show()