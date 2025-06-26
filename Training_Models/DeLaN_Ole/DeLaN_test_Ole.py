'''
Autor:      Ole Uphaus
Datum:     25.04.2025
Beschreibung:
Dieses Skript soll das von mir erstellte Deep Lagrangian Network testen. Dabei werden Frei erfundene Inputs durch das Netzwerk geschickt und die entsprechenden Outputs untersucht. Damit lassen sich vor allem die Matrixoperationen und Dimensionen nachvollziehen.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from DeLaN_model_Ole import Deep_Lagrangian_Network

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Seed setzen für Reproduzierbarkeit
seed = 41
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

hyper_param = {
    # Netzparameter
    'hidden_width': 64,
    'hidden_depth': 2,
    'activation_fnc': 'elu',
    'activation_fnc_diag': 'relu',

    # Initialisierung
    'bias_init_constant': 1.e-3,
    'wheight_init': 'xavier_normal',

    # Lagrange Dynamik
    'L_diagonal_offset': 1.e-3,
    
    # Training
    'dropuot': 0.0,
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-4,
    'n_epoch': 500,

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
    'save_model': False}

# Testnetzwerk erstellen
n_dof = 2   # Später aus Trainingsdaten auslesen
test_net = Deep_Lagrangian_Network(n_dof, **hyper_param)

# Netzwerk erproben
test_q = torch.tensor([[1, 3], [2, 1], [3, 2]], dtype=torch.float32)
test_qd = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.float32)
test_qdd = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.float32)
test_tau = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.float32)
print('Test Input q: \n', test_q)
print('Test Input qd: \n', test_qd)
print('Test Input qdd: \n', test_qdd)
print()
# Inverse Dynamik auswerten
output_inv = test_net(test_q, test_qd, test_qdd)

# Outputs ansehen
print('Massenmatrix H: \n', output_inv[1])
print()
print('Corioliskräfte c: \n', output_inv[2])
print()
print('T_dt: \n', output_inv[6])
print()
print('V_dt: \n', output_inv[7])
print()
print('tau_fric: \n', output_inv[4])
print()

# # Vorwärts Dynamik auswerten
# output_for = test_net.forward_dynamics(test_q, test_qd, test_tau)

# # Outputs ansehen
# print('Massenmatrix H: \n', output_for[1])
# print()
# print('Corioliskräfte c: \n', output_for[2])
# print()
# print('q_pp: \n', output_for[0])
# print()
