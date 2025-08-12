'''
Autor:     Ole Uphaus
Datum:     06.06.2025
Beschreibung:
Dieses Skript dient dazu, die Trainingsdaten plotten zu können, om die Intervalle der Trajektorien nachvollziehen zu können.
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

# Trainings- und Testdaten laden
target_folder = 'Studienarbeit_Data' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen', 'Studienarbeit_Data'
features_training1, _, _, _, _ = extract_training_data('Variation_Datenmenge_Nruns_10.mat', target_folder)
features_training2, _, _, _, _ = extract_training_data('Variation_Datenmenge_Nruns_50.mat', target_folder)

# Anzahl der Samples 
samples_training1 = np.arange(1, features_training1.shape[0] + 1)
samples_training2 = np.arange(1, features_training2.shape[0] + 1)

# Plots für Studienarbeit
plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\05_Training_Regelung'
plot_1_name = os.path.join(plot_path, 'Abbildung_Variation_Datenanzahl.pdf')

# Plot 1
double_subplot_varx(
    [samples_training1, samples_training2],
    [features_training1[:, 0].reshape(-1, 1), 
     features_training2[:, 0].reshape(-1, 1)],
    [r'', r'Datenpunktindex'],
    [r'$r_{RS} \, / \, \mathrm{m}$', r'$r_{RS} \, / \, \mathrm{m}$'],
    [r'$N_r = 8$', r'$N_r = 40$'],
    [[r'$r_{d,RS}$', r'$r_{RS}$'], [r'$\varphi_{d,RS}$', r'$\varphi_{RS}$']],
    plot_1_name,
    True,
    False
)
