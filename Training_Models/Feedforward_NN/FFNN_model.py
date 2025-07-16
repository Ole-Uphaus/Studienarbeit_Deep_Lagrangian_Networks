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

        # Aktivierungsfunktion festlegen
        self.activation_fnc = self.get_activation_fnc(hyper_param['activation_fnc'])

        # Liste mit allen Schichten
        layers = []

        # Input Schicht
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(self.activation_fnc)
        layers.append(nn.Dropout(self.dropout))

        # Hidden Layers als Schleife anhängen
        for i in range(self.depth - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation_fnc)
            layers.append(nn.Dropout(self.dropout))

        # Output Layer hinzufügen
        layers.append(nn.Linear(self.hidden_size, self.n_dof))

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(*layers)

    def get_activation_fnc(self, name):
        # Alles klein geschrieben
        name = name.lower()

        # Alle erlaubten Aktivierungsfunktionen druchgehen (hier noch weitere hinzufügen)
        if name == 'relu':
            activation_fnc = nn.ReLU()
        elif name == 'softplus':
            activation_fnc = nn.Softplus()
        elif name == 'tanh':
            activation_fnc = nn.Tanh()
        elif name == 'elu':
            activation_fnc = nn.ELU()
        elif name == 'gelu':
            activation_fnc = nn.GELU()
        elif name == 'silu':
            activation_fnc = nn.SiLU()
        elif name == 'mish':
            activation_fnc = nn.Mish()
        else:
            activation_fnc = nn.ReLU()

        return activation_fnc
    
    def forward(self, x):
        return self.net(x)