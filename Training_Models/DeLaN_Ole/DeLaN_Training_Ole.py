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

from DeLaN_model_Ole import Intern_NN

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Parameter festlegen
hyper_param = {
    'hidden_width': 64,
    'hidden_depth': 2,
    'activation_fnc': 'relu',
    'batch_size': 512,
    'learning_rate': 1.e-5,
    'n_epoch': 2000,
    'save_model': False}

# Testnetzwerk erstellen
n_dof = 2   # Später aus Trainingsdaten auslesen
test_net = Intern_NN(n_dof, **hyper_param)

# Netzwerk erproben
test_input = torch.tensor([[1, 3], [2, 4], [3, 2], [5, 2], [3, 3]], dtype=torch.float32)
print('Test Input: \n', test_input)
print()
output = test_net(test_input)

print('Test Output g: \n', output[0])
print()
print('Test Output Hauptdiagonale: \n', output[1])
print()
print('Test Output Nebendiagonale: \n', output[2])