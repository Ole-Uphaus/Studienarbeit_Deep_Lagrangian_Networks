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
import os

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
    features = data['features']
    labels = data['labels']

    return features, labels

# Erstellung neuronales Netz (Klasse)
class Feed_forward_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feed_forward_NN, self).__init__()

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)


# Trainingsdaten laden
features, labels = extract_training_data('SimData__2025_03_20_13_59_55.mat')

# Trainingsdaten in Torch-Tensoren umwandeln
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Dataset und Dataloader erstellen
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Neuronales Netz initialisieren
input_size = features.shape[1]
output_size = labels.shape[1]
hidden_size = 10

model = Feed_forward_NN(input_size, hidden_size, output_size)


