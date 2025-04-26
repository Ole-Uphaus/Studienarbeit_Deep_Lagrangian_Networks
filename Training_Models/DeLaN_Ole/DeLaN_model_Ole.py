'''
Autor:      Ole Uphaus
Datum:     25.04.2025
Beschreibung:
Dieses Skript beinhaltet das pytorch Modell des Deep Lagrangian Networks. Das Modell orientiert sich am Paper von Michael Lutter, wird sich in der Implementierung jedoch von seinem Code unterscheiden. 
'''

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

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
        n_tril = int(n_dof*(n_dof - 1)/2)
        self.output_L_tril = nn.Linear(hyper_param['hidden_width'], n_tril)

    def get_activation_fnc(self, name):
        # Alles klein geschrieben
        name = name.lower()

        # Alle erlaubten Aktivierungsfunktionen druchgehen (hier noch weitere hinzufügen)
        if name == 'relu':
            activation_fnc = nn.ReLU()
        else:
            activation_fnc = nn.ReLU()

        return activation_fnc
    
    def forward(self, q):
        # Netzwerkeingang q iterativ durch alle Layer geben
        for layer in self.layers:
            q = self.activation_fnc(layer(q))

        # Jeweils die Netzwerk Outputs einzeln berechnen und zurückgeben
        return self.output_g(q), self.ReLu(self.output_L_diag(q)), self.output_L_tril(q)
    
    
class Deep_Lagrangian_Network(nn.Module):
    def __init__(self, n_dof, **hyper_param):
        super(Deep_Lagrangian_Network, self).__init__()

        # Internes Netz definieren
        self.Intern_NN = Intern_NN(n_dof, **hyper_param)

        # Parameter festsetzen
        self.n_dof = n_dof

    def forward(self, q, qd, qdd):

        return self.lagrangian_dynamics(q, qd, qdd)

    def lagrangian_dynamics(self, q, qd, qdd):

        # Eingänge q reshapen, damit Ausgangsdimension stimmt
        q = q.view((-1, self.n_dof))

        # Batch Größe bestimmen
        batch_size = q.shape[0]

        # Internes Netz mit Eingangswerten (q) auswerten
        output_g, output_L_diag, output_L_tril = self.Intern_NN(q)

        # Partielle Ableitungen der Einträge in L bezüglich der Eingänge (q) berechnen
        output_L_diag_dq = torch.zeros((batch_size, output_L_diag.shape[1], self.n_dof))
        output_L_tril_dq = torch.zeros((batch_size, output_L_tril.shape[1], self.n_dof))

        for i in range(batch_size): # Schleife, damit nicht nach allen Eingängen des Batches abgeleitet wird
            output_L_diag_dq[i] = jacobian(lambda inp: self.Intern_NN(inp)[1], q[i, :], create_graph=True)    # Ableitung Diagonalelemente nach q
            output_L_tril_dq[i] = jacobian(lambda inp: self.Intern_NN(inp)[2], q[i, :], create_graph=True)    # Ableitung Nebendiaginalelemente (untere Dreiecksmatrix) nach q

        return output_L_diag, output_L_tril, output_L_diag_dq, output_L_tril_dq

       