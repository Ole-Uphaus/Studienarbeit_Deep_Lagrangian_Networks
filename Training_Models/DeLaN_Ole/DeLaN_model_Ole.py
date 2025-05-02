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
        self.L_diagonal_offset = hyper_param['L_diagonal_offset']

    def forward(self, q, qd, qdd):

        return self.lagrangian_dynamics(q, qd, qdd)

    def lagrangian_dynamics(self, q, qd, qdd):

        # Eingänge q reshapen, damit Ausgangsdimension stimmt
        q = q.view((-1, self.n_dof))    # q.shape = (batch_size, 1)
        qd = qd.view((-1, self.n_dof))
        qdd = qdd.view((-1, self.n_dof))

        # Batch Größe bestimmen
        self.batch_size = q.shape[0]

        # Internes Netz mit Eingangswerten (q) auswerten
        output_g, output_L_diag, output_L_tril = self.Intern_NN(q)  # output_L_diag.shape = (batch_size, n_dof), output_L_tril.shape = (batch.size, anz_elemente_unter_hauptdiagonalen)

        # Partielle Ableitungen der Einträge in L bezüglich der Eingänge (q) berechnen
        output_L_diag_dq = torch.zeros((self.batch_size, output_L_diag.shape[1], self.n_dof))
        output_L_tril_dq = torch.zeros((self.batch_size, output_L_tril.shape[1], self.n_dof))

        for i in range(self.batch_size): # Schleife, damit nicht immer nach allen Eingängen des Batches abgeleitet wird
            output_L_diag_dq[i] = jacobian(lambda inp: self.Intern_NN(inp)[1], q[i, :], create_graph=True)    # Ableitung Diagonalelemente von L nach q (output_L_diag_dq.shape = (batch_size, n_dof, n_dof))
            output_L_tril_dq[i] = jacobian(lambda inp: self.Intern_NN(inp)[2], q[i, :], create_graph=True)    # Ableitung Nebendiaginalelemente (untere Dreiecksmatrix) von L nach q (output_L_diag_dq.shape = (batch_size, anz_elemente_unter_hauptdiagonalen, n_dof))

        # L zusammensetzen
        L = self.construct_L_or_L_dq(output_L_diag, output_L_tril)  # (L.shape = (batch_size, n_dof, n_dof))
        L_transp = L.transpose(1, 2)    # Dimensionen 1 und 2 vertauschen (L_transp.shape = (batch_size, n_dof, n_dof)

        # L_dq zusammensetzen
        L_dq = self.construct_L_or_L_dq(output_L_diag_dq, output_L_tril_dq) # L_dq.shape(batch_size, n_dof, n_dof, n_dof)
        L_dq_transpose = L_dq.transpose(1, 2)

        # Massenmatrix H berechnen (L * LT)
        H = torch.matmul(L, L_transp)   # H.shape = (batch_size, n_dof, n_dof)

        # L_dt berechnen (L_dq * qd)    
        L_dt = torch.einsum('bijk,bk->bij', L_dq, qd)    # L_dt.shape = (batch_size, n_dof, n_dof)
        L_dt_transpose = L_dt.transpose(1, 2)

        # H_dt berechnen (L * L_dtT + L_dt * LT)
        H_dt = torch.matmul(L, L_dt_transpose) + torch.matmul(L_dt, L_transp)   # H_dt.shape = (batch_size, n_dof, n_dof)

        # H_dq berechnen (L_dq * LT + L * L_dqT)
        # H_dq = torch.matmul(L_dq, L_transp.unsqueeze(-1))


        return output_L_diag, output_L_tril, output_L_diag_dq, output_L_tril_dq, L, L_dq, L_transp, L_dq_transpose, H, L_dt, L_dt_transpose
    
    def construct_L_or_L_dq(self, L_diag, L_tril):
        # benötigte Dimensionen von L herausfinden (unterscheiden ob L oder L_dq zusammengesetzt werden soll)
        if len(L_diag.shape) == 2:
            dim = (self.batch_size, self.n_dof, self.n_dof)

            # Einheitsmatrix mit diagonalem Offset auf der Hauptdiagonalen
            diagonal_offset = torch.eye(self.n_dof) * self.L_diagonal_offset
        elif len(L_diag.shape) == 3:
            dim = (self.batch_size, self.n_dof, self.n_dof, self.n_dof)

            # diagonele Offsetmatrix mit richtigen Dimensionen als Nullmatrix initialisieren
            diagonal_offset = torch.zeros((self.n_dof, self.n_dof, self.n_dof), dtype=L_diag.dtype)

        # Leere Matrix erstellen
        L = torch.zeros(dim, dtype=L_diag.dtype)

        # Indizees Hauptdiagonale
        idx_main = range(self.n_dof)

        # Indizees der Elemente unter der Hauptdiagonalen
        rows, cols = torch.tril_indices(self.n_dof, self.n_dof, offset=-1)

        # L Zusammensetzen
        for i in range(self.batch_size):
            L[i, idx_main, idx_main] = L_diag[i]   # Hauptdiagonale
            L[i, rows, cols] = L_tril[i]    # Elemente unter der Hauptdiagonalen

            # Diagonalen Offset hinzufügen (Nullmatrix wenn dL/dq)
            L[i] += diagonal_offset

        return L

       