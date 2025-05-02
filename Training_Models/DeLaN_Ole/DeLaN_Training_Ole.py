'''
Autor:      Ole Uphaus
Datum:     25.04.2025
Beschreibung:
Dieses Skript soll das von mir erstellte Neep Lagrangian Network trainieren. Alle benötigten Funktionen und Klassen werden extern eingebunden. Alle Parameter und Hyperparameter des Netzwerk werden in diesem Skript festgelegt.
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

# Parameter festlegen
hyper_param = {
    'hidden_width': 64,
    'hidden_depth': 2,
    'L_diagonal_offset': 0.1,
    'activation_fnc': 'relu',
    'batch_size': 512,
    'learning_rate': 1.e-5,
    'n_epoch': 2000,
    'save_model': False}

# Testnetzwerk erstellen
n_dof = 2   # Später aus Trainingsdaten auslesen
test_net = Deep_Lagrangian_Network(n_dof, **hyper_param)

# Netzwerk erproben
test_q = torch.tensor([[1, 3], [2, 1], [3, 2]], dtype=torch.float32)
test_qd = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.float32)
print('Test Input q: \n', test_q)
print('Test Input qd: \n', test_qd)
print()
output = test_net(test_q, test_qd, test_q)

# print('Test Output Diagonalelemente von L: \n', output[0])
# print()
# print('Test Output Untere Dreiecksmatrix Einträge: \n', output[1])
# print()
# print('Test Output Ableitung Diagonalelemente von L: \n', output[2])
# print()
# print('Test Output Ableitung Untere Dreiecksmatrix Einträge von L: \n', output[3])
# print()
print('Matrix L: \n', output[4])
print()
print('Ableitung dL(dq): \n', output[5])
# print()
# print('Matrix L_Transp: \n', output[6])
# print()
# print('Matrix L_dqT: \n', output[7])
# print()
# print('Matrixmultiplikation L*LT (H): \n', output[8])
print()
print('Matrixmultiplikation L_dq*qd (L_dt): \n', output[9])
# print()
# print('L_dt transponiert: \n', output[10])
# print()
# print('H_dt: \n', output[11])

