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
target_folder = 'Torsionsschwinger_Messungen' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen'
features_training, labels_training, features_test, labels_test, Mass_Cor_test = extract_training_data('Measuring_data_Training_Torsionsschwinger.mat', target_folder)

# Anzahl der Samples 
samples_training = np.arange(1, features_training.shape[0] + 1)
samples_test = np.arange(1, features_test.shape[0] + 1)

# Trainingsdaten plotten
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(samples_training, features_training[:, 0])
plt.title('q1')
plt.xlabel('Samples')
plt.ylabel('q1')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(samples_training, features_training[:, 1])
plt.title('q2')
plt.xlabel('Samples')
plt.ylabel('q2')
plt.grid(True)

plt.tight_layout()

# Testdaten plotten
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(samples_test, features_test[:, 0])
plt.title('q1')
plt.xlabel('Samples')
plt.ylabel('q1')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(samples_test, features_test[:, 1])
plt.title('q2')
plt.xlabel('Samples')
plt.ylabel('q2')
plt.grid(True)

plt.tight_layout()

plt.show()
