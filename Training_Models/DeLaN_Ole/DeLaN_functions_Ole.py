'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
Dieses Skript enthält Funktionen im Zusammenhang mit dem Training und der Erprobung von Deep Lagrangien Networks.
'''

import scipy.io
import os
import numpy as np

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