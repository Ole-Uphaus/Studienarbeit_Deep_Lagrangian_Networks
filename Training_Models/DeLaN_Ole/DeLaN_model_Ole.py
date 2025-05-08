'''
Autor:      Ole Uphaus
Datum:     25.04.2025
Beschreibung:
Dieses Skript beinhaltet das pytorch Modell des Deep Lagrangian Networks. Das Modell orientiert sich am Paper von Michael Lutter, wird sich in der Implementierung jedoch von seinem Code unterscheiden. 
'''

import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.init as init

class Intern_NN(nn.Module):
    def __init__(self, n_dof, **hyper_param):
        super(Intern_NN, self).__init__()
        # Parameter festsetzen
        self.bias_init_constant = hyper_param['bias_init_constant']

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

        # Netzgewichte Initialisieren (Xavier Normal)
        self.init_weights()

    def get_activation_fnc(self, name):
        # Alles klein geschrieben
        name = name.lower()

        # Alle erlaubten Aktivierungsfunktionen druchgehen (hier noch weitere hinzufügen)
        if name == 'relu':
            activation_fnc = nn.ReLU()
        elif name == 'softplus':
            activation_fnc = nn.Softplus()
        else:
            activation_fnc = nn.ReLU()

        return activation_fnc
    
    def init_weights(self):
        # Initialisiere alle linearen Layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)  # Initialisierung der gewichte
                init.constant_(m.bias, self.bias_init_constant)          # Initialisierung des Bias
    
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
        # Device festlegen
        self.device = q.device

        # Eingänge q reshapen, damit Ausgangsdimension stimmt
        q = q.view((-1, self.n_dof))    # q.shape = (batch_size, 1)
        qd = qd.view((-1, self.n_dof))
        qdd = qdd.view((-1, self.n_dof))

        # Batch Größe bestimmen
        self.batch_size = q.shape[0]

        # Autograd verfolgung für q aktivieren
        q.requires_grad_(True)

        # Internes Netz mit Eingangswerten (q) auswerten
        output_g, output_L_diag, output_L_tril = self.Intern_NN(q)  # output_L_diag.shape = (batch_size, n_dof), output_L_tril.shape = (batch.size, anz_elemente_unter_hauptdiagonalen)

        # Partielle Ableitungen der Einträge in L bezüglich der Eingänge (q) berechnen
        output_L_diag_dq = self.compute_Jacobian_batched(output_L_diag, q)
        output_L_tril_dq = self.compute_Jacobian_batched(output_L_tril, q)

        # L zusammensetzen
        L = self.construct_L_or_L_dq(output_L_diag, output_L_tril)  # (L.shape = (batch_size, n_dof, n_dof))
        L_transp = L.transpose(1, 2)    # Dimensionen 1 und 2 vertauschen (L_transp.shape = (batch_size, n_dof, n_dof)

        # L_dq zusammensetzen
        L_dq = self.construct_L_or_L_dq(output_L_diag_dq, output_L_tril_dq) # L_dq.shape(batch_size, n_dof, n_dof, n_dof)
        L_dq_transpose = L_dq.transpose(1, 2)

        # Massenmatrix H berechnen (L * LT)
        H = torch.einsum('bij,bjk->bik', L, L_transp)   # H.shape = (batch_size, n_dof, n_dof)

        # L_dt berechnen (L_dq * qd)    
        L_dt = torch.einsum('bijc,bc->bij', L_dq, qd)    # L_dt.shape = (batch_size, n_dof, n_dof)
        L_dt_transpose = L_dt.transpose(1, 2)

        # H_dt berechnen (L * L_dtT + L_dt * LT)
        H_dt = torch.einsum('bij,bjk->bik', L, L_dt_transpose) + torch.einsum('bij,bjk->bik', L_dt, L_transp)   # H_dt.shape = (batch_size, n_dof, n_dof)

        # H_dq berechnen (L_dq * LT + L * L_dqT)
        H_dq = torch.einsum('bijc,bjk->bikc', L_dq, L_transp) + torch.einsum('bij,bjkc->bikc', L, L_dq_transpose)   # H_dq.shape(batch_size, n_dof, n_dof, n_dof)

        # qdT_H_dq_qd berechnen (qdT * H_dq * qd)
        H_dq_qd = torch.einsum('bijc,bc->bij', H_dq, qd)    # H_dq_qd.shape(batch_size, n_dof, n_dof)
        qdT_H_dq_qd = torch.einsum('bi,bij->bj', qd, H_dq_qd)   # qdT_H_dq_qd.shape(batch_size, n_dof)

        # Coriolisterme berechnen (H_dt * qd - 0.5 * qdT_H_dq_qd)
        c = torch.einsum('bij,bj->bi', H_dt, qd) - 0.5 * qdT_H_dq_qd    # c.shape(batch_size, n_dof)

        # Inverse Dynamik auswerten
        tau = torch.einsum('bij,bj->bi', H, qdd) + c + output_g

        return tau, H, c, output_g
    
    def compute_Jacobian_batched(self, output_L, input_q):
        # Dimensionen des Outputs bekommen
        output_L_dim = output_L.shape[1]

        # Jacobimatrix initialisieren
        jac = torch.zeros((self.batch_size, output_L_dim, self.n_dof), device=self.device, dtype=output_L.dtype)

        for i in range(output_L_dim):
            # Gradienten Batchweise für einen Output berechnen (Summierung ist zulässig, da ja nur das eine Element in der SUmme vom jeweiligen Output abhängt und der Rest ja null ist)
            gradient = grad(output_L[:, i].sum(), input_q, retain_graph=True, create_graph=True, allow_unused=False)[0]

            # Entsprechende Einträge in Jacobimatrix ergänzen
            jac[:, i, :] = gradient

        return jac
    
    def construct_L_or_L_dq(self, L_diag, L_tril):
        # benötigte Dimensionen von L herausfinden (unterscheiden ob L oder L_dq zusammengesetzt werden soll)
        if len(L_diag.shape) == 2:
            dim = (self.batch_size, self.n_dof, self.n_dof)

            # Einheitsmatrix mit diagonalem Offset auf der Hauptdiagonalen
            diagonal_offset = torch.eye(self.n_dof, device=self.device) * self.L_diagonal_offset
        elif len(L_diag.shape) == 3:
            dim = (self.batch_size, self.n_dof, self.n_dof, self.n_dof)

            # diagonele Offsetmatrix mit richtigen Dimensionen als Nullmatrix initialisieren
            diagonal_offset = torch.zeros((self.n_dof, self.n_dof, self.n_dof), dtype=L_diag.dtype, device=self.device)

        # Leere Matrix erstellen
        L = torch.zeros(dim, dtype=L_diag.dtype, device=self.device)

        # Indizees Hauptdiagonale
        idx_main = range(self.n_dof)

        # Indizees der Elemente unter der Hauptdiagonalen
        rows, cols = torch.tril_indices(self.n_dof, self.n_dof, offset=-1, device=self.device)

        # L Zusammensetzen
        L[:, idx_main, idx_main] = L_diag   # Hauptdiagonale
        L[:, rows, cols] = L_tril    # Elemente unter der Hauptdiagonalen

        # Diagonalen Offset hinzufügen (Nullmatrix wenn dL/dq)
        L += diagonal_offset

        return L

       