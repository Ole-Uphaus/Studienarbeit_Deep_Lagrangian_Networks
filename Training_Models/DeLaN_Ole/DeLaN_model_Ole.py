'''
Autor:      Ole Uphaus
Datum:     25.04.2025
Beschreibung:
Dieses Skript beinhaltet das pytorch Modell des Deep Lagrangian Networks. Das Modell orientiert sich am Paper von Michael Lutter, wird sich in der Implementierung jedoch von seinem Code unterscheiden. 
'''

import torch
import torch.nn as nn

class Intern_NN(nn.Module):
    def __init__(self, n_dof, **hyper_param):
        super(Intern_NN, self).__init__()

        # Aktivierungsfunktion festlegen
        self.activation_fnc = self.get_activation_fnc(hyper_param['activation_fnc'])
        self.ReLu = nn.ReLU()   # Aktivierungsfunktion für Output

        # Liste mit allen Layern
        self.layers = nn.ModuleList()

        # Eingangs Layer
        self.layers.append(nn.Linear(n_dof, hyper_param['hidden_width']))

        # Hidden Layer
        for _ in range(hyper_param['hidden_depth'] - 1):
            self.layers.append(nn.Linear(hyper_param['hidden_width'], hyper_param['hidden_width']))

        # Output Gravitationsterme - Dimension==n_dof (Linearer Layer) 
        self.output_g = nn.Linear(hyper_param['hidden_width'], n_dof)

        # Output Hauptdiagonalelemente von L - Dimension==n_dof (ReLu Layer) 
        self.output_L_diag = nn.Linear(hyper_param['hidden_width'], n_dof)

        # Output Nebendiagonalelemente (untere Dreiecksmatrix) von L
        n_side_diag = int(n_dof*(n_dof - 1)/2)
        self.output_L_side_diag = nn.Linear(hyper_param['hidden_width'], n_side_diag)

    def get_activation_fnc(self, name):
        # Alles klein geschrieben
        name = name.lower()

        # Alle erlaubten Aktivierungsfunktionen druchgehen (hier noch weitere hinzufügen)
        if name == 'relu':
            activation_fnc = nn.ReLU()
        else:
            activation_fnc = nn.ReLU()

        return activation_fnc
    
    def forward(self, x):
        # Netzwerkeingang x iterativ durch alle Layer geben
        for layer in self.layers:
            x = self.activation_fnc(layer(x))

        # Jeweils die Netzwerk Outputs einzeln berechnen und zurückgeben
        return self.output_g(x), self.ReLu(self.output_L_diag(x)), self.output_L_side_diag(x)