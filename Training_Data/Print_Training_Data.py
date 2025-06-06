'''
Autor:     Ole Uphaus
Datum:     06.06.2025
Beschreibung:
Dieses Skript dient dazu, die Trainingsdaten plotten zu können, om die Intervalle der Trajektorien nachvollziehen zu können.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

def extract_training_data(file_name):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, 'MATLAB_Simulation', file_name)

    # Pfad normieren
    data_path = os.path.normpath(data_path)

    # Daten extrahieren
    data = scipy.io.loadmat(data_path)

    features_training = data['features_training']
    labels_training = data['labels_training']
    features_test = data['features_test']
    labels_test = data['labels_test']
    Mass_Cor_test = data['Mass_Cor_test']

    # Zusammensetzung der vektoren ändern, da Erstellung in Matlab für Inverse Dynamik ausgelegt war
    features_training_delan = np.concatenate((features_training[:, :4], labels_training), axis=1)   # (q, qp, qpp)
    features_test_delan = np.concatenate((features_test[:, :4], labels_test), axis=1)   

    labels_training_delan = features_training[:, 4:]
    labels_test_delan = features_test[:, 4:]

    return features_training_delan, labels_training_delan, features_test_delan, labels_test_delan, Mass_Cor_test

# Trainings- und Testdaten laden 
features_training, labels_training, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat')  # Mein Modell

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
