'''
Autor:      Ole Uphaus
Datum:     10.08.2025
Beschreibung:
Dieses Skript enthält einen Plot mit gängigen Aktivierungsfunktionen für die Studienarbeit.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
DeLaN_dir_path = os.path.join(script_path, '..', 'Training_Models', 'DeLaN_Ole')

if DeLaN_dir_path not in sys.path:
    sys.path.insert(0, DeLaN_dir_path)

from DeLaN_functions_Ole import *

# Aktivierungsfunktionen definieren
def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def tanh(x):
    return np.tanh(x)

def softplus(x):
    return np.log1p(np.exp(x))  # log(1 + exp(x)) stabil berechnet

# Eingabewerte
x = np.linspace(-5, 5, 500)

# Funktionen berechnen
y_relu = relu(x)
y_elu = elu(x)
y_tanh = tanh(x)
y_softplus = softplus(x)

# Speicherpfad
plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\02_Grundlagen'
plot_1_name = os.path.join(plot_path, 'Plot_Aktivierungsfunktionen.pdf')

# Plot Aktivierungsfunktion
quad_subplot(
    x,
    [y_relu.reshape(-1, 1), 
     y_elu.reshape(-1, 1),
     y_tanh.reshape(-1, 1),
     y_softplus.reshape(-1, 1)],
    r'x',
    [r'$f(x)$', r'$f(x)$', r'$f(x)$', r'$f(x)$'],
    ['ReLU', 'ELU', 'tanh', 'Softplus'],
    [['DeLaN', 'GT'], ['DeLaN', 'GT']],
    plot_1_name,
    True,
    False
)

