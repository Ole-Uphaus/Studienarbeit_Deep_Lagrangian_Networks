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
    'hidden_width': 64,
    'hidden_depth': 2,
    'L_diagonal_offset': 1.e-2,
    'activation_fnc': 'softplus',
    'dropuot': 0.0,
    'bias_init_constant': 1.e-2,
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-4,
    'n_epoch': 2000,
    'save_model': False}

# Testnetzwerk erstellen
n_dof = 2   # Später aus Trainingsdaten auslesen
test_net = Deep_Lagrangian_Network(n_dof, **hyper_param)

# Netzwerk erproben
test_q = torch.tensor([[1, 3], [2, 1], [3, 2]], dtype=torch.float32)
test_qd = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.float32)
test_qdd = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.float32)
print('Test Input q: \n', test_q)
print('Test Input qd: \n', test_qd)
print('Test Input qdd: \n', test_qdd)
print()
output = test_net(test_q, test_qd, test_qdd)

# Outputs ansehen
print('Massenmatrix H: \n', output[1])
print()
print('Corioliskräfte c: \n', output[2])
print()
