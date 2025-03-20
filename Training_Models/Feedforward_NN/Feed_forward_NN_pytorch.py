'''
Autor:      Ole Uphaus
Datum:      20.03.2025
Beschreibung:
Dieses Skript soll auf Grundlage der Simuleirten Trainingsdaten ein neuronales Netz trainieren. Dies ist jedoch nur ein Feed-Forward Netz, das keine Informationen über lagrange Gleichungen enthält. Die benötigten Trainingsdaten werden aus einem .mat File extrahiert.
'''

import scipy.io
import torch
import os

def extract_training_data(file_name):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', 'MATLAB_Simulation', file_name)

    # Pfad normieren
    data_path = os.path.normpath(data_path)

    # Daten extrahieren
    data = scipy.io.loadmat(data_path)
    features = data['features']
    labels = data['labels']

    return features, labels

# Trainingsdaten laden
features, labels = extract_training_data('SimData__2025_03_20_13_59_55.mat')


