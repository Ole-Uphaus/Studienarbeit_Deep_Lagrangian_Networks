'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
Dieses Skript enthält Funktionen im Zusammenhang mit dem Training und der Erprobung von Deep Lagrangien Networks.
'''

import scipy.io
import os
import numpy as np

def extract_training_data(file_name, target_folder):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', target_folder, file_name)

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

def model_evaluation(model, q_test, qd_test, qdd_test, tau_test):
    # Forward pass
    out_eval = model(q_test, qd_test, qdd_test)    # Inverses Modell
    qdd_hat, _, _, _ = model.forward_dynamics(q_test, qd_test, tau_test) # Vorwärts Modell

    tau_hat_eval = out_eval[0].cpu().detach().numpy()   # Tesnoren auf cpu legen, gradienten entfernen, un numpy arrays umwandeln
    H_eval = out_eval[1].cpu().detach().numpy()
    c_eval = out_eval[2].cpu().detach().numpy()
    g_eval = out_eval[3].cpu().detach().numpy()
    tau_fric_eval = out_eval[4].cpu().detach().numpy()

    # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
    err_inv_dyn_test = np.sum((tau_hat_eval - tau_test.cpu().detach().numpy())**2, axis=1)
    mean_err_inv_dyn_eval = np.mean(err_inv_dyn_test)

    # Fehler aus Vorwärtsmodell berechnen (Schätzung von qdd)
    err_for_dyn_test = np.sum((qdd_hat.cpu().detach().numpy() - qdd_test.cpu().detach().numpy())**2, axis=1)
    mean_err_for_dyn_eval = np.mean(err_for_dyn_test)

    # Test Loss berechnen
    test_loss = mean_err_inv_dyn_eval + mean_err_for_dyn_eval

    return test_loss, tau_hat_eval, H_eval, c_eval, g_eval, tau_fric_eval