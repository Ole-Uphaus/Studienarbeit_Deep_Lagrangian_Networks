'''
Autor:      Ole Uphaus
Datum:     16.07.2025
Beschreibung:
In diesem Skript wird das Feed-Forward Modell zum Vergleich mit dem DeLaN Modell erstellt.
'''

import torch.nn as nn

# Erstellung neuronales Netz (Klasse)
class Feed_forward_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, depth):
        super(Feed_forward_NN, self).__init__()

        layers = []

        # Input Schicht
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden Layers als Schleife anhängen
        for i in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output Layer hinzufügen
        layers.append(nn.Linear(hidden_size, output_size))

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)