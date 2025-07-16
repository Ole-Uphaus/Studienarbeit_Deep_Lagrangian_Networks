'''
Autor:      Ole Uphaus
Datum:     16.07.2025
Beschreibung:
In diesem Skript wird das Feed-Forward Modell zum Vergleich mit dem DeLaN Modell erstellt.
'''

import torch.nn as nn

# Erstellung neuronales Netz (Klasse)
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, n_dof, **hyper_param):
        super(FeedForwardNN, self).__init__()

        # Parameter auslesen
        self.input_size = input_size
        self.n_dof = n_dof
        self.hidden_size = hyper_param['hidden_size']
        self.depth = hyper_param['depth']
        self.dropout = hyper_param['dropout']

        # Liste mit allen Schichten
        layers = []

        # Input Schicht
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))

        # Hidden Layers als Schleife anhängen
        for i in range(self.depth - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        # Output Layer hinzufügen
        layers.append(nn.Linear(self.hidden_size, self.n_dof))

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)