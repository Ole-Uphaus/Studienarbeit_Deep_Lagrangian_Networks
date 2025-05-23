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

from DeLaN_model_damping_Ole import Deep_Lagrangian_Network

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
    'hidden_width': 32,
    'hidden_depth': 3,
    'activation_fnc': 'elu',

    # Initialisierung
    'bias_init_constant': 1.e-3,
    'wheight_init': 'xavier_normal',

    # Lagrange Dynamik
    'L_diagonal_offset': 1.e-4,
    
    # Training
    'dropuot': 0.0,
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-4,
    'n_epoch': 2000,

    # Sonstiges
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
# # Inverse Dynamik auswerten
# output_inv = test_net(test_q, test_qd, test_qdd)

# # Outputs ansehen
# print('Massenmatrix H: \n', output_inv[1])
# print()
# print('Corioliskräfte c: \n', output_inv[2])
# print()

# Vorwärts Dynamik auswerten
output_for = test_net.forward_dynamics(test_q, test_qd, test_tau)

# Outputs ansehen
print('Massenmatrix H: \n', output_for[1])
print()
print('Corioliskräfte c: \n', output_for[2])
print()
print('q_pp: \n', output_for[0])
print()
