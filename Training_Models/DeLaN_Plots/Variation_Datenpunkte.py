'''
Autor:      Ole Uphaus
Datum:     29.07.2025
Beschreibung:
In diesem Skript wird das DeLaN modell mit variablen Datensätzen trainiert um diesen Einfluss zu untersuchen.
'''

import os
import sys

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
DeLaN_dir_path = os.path.join(script_path, '..', 'DeLaN_Ole')

if DeLaN_dir_path not in sys.path:
    sys.path.insert(0, DeLaN_dir_path)

from DeLaN_functions_Ole import *
from DeLaN_training_Ole import *

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Seed setzen für Reproduzierbarkeit
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Trainings- und Testdaten laden
target_folder = 'Studienarbeit_Data' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen' 'Studienarbeit_Data'
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Auswählen, ob nur geplottet oder auch trainiert werden soll
just_plot = True

if just_plot == False:

    ###################### Wenig Daten ######################

    # Parameter festlegen
    hyper_param = {
        # Netzparameter
        'hidden_width': 64,
        'hidden_depth': 2,
        'activation_fnc': 'elu',
        'activation_fnc_diag': 'softplus',

        # Initialisierung
        'bias_init_constant': 1.e-3,
        'wheight_init': 'xavier_normal',

        # Lagrange Dynamik
        'L_diagonal_offset': 1.e-2,
        
        # Training
        'dropuot': 0.0,
        'batch_size': 512,
        'learning_rate': 5.e-4,
        'weight_decay': 1.e-4,
        'n_epoch': 10000,

        # Reibungsmodell
        'use_friction_model': False,
        'friction_model_init_d': [0.01, 0.01],
        'friction_model_init_c': [0.01, 0.01],
        'friction_model_init_s': [0.01, 0.01],
        'friction_model_init_v': [0.01, 0.01],
        'friction_epsilon': 100.0,

        # Sonstiges
        'use_inverse_model': True,
        'use_forward_model': True,
        'use_energy_consumption': False,
        'save_model': False}

    # Trainings- und Testdaten laden
    features_training, labels_training, _, _, _ = extract_training_data('Variation_Datenmenge_Nruns_10.mat', target_folder)  # Mein Modell Trainingsdaten

    # Modell trainieren
    DeLaN_network, results = Delan_Train_Eval(
            target_folder,
            features_training,
            labels_training,
            features_test,
            labels_test,
            hyper_param,
            device
        )

    # Ergebnisse entpacken
    tau_hat_test_Nruns_10 = results['tau_hat_test']
    rmse_tau_percent_Nruns_10 = results['rmse_tau_percent']

    ###################### Viele Daten ######################
    features_training, labels_training, _, _, _ = extract_training_data('Variation_Datenmenge_Nruns_50.mat', target_folder)  # Mein Modell Trainingsdaten
    hyper_param['n_epoch'] = 2000

    # Modell trainieren
    DeLaN_network, results = Delan_Train_Eval(
            target_folder,
            features_training,
            labels_training,
            features_test,
            labels_test,
            hyper_param,
            device
        )

    # Ergebnisse entpacken
    tau_hat_test_Nruns_50 = results['tau_hat_test']
    rmse_tau_percent_Nruns_50 = results['rmse_tau_percent']

    # Ergebnisse plotten
    samples_vec = np.arange(1, tau_hat_test_Nruns_50.shape[0] + 1).reshape(-1, 1)

    plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
    plot_1_name = os.path.join(plot_path, 'Variation_Datenpunkte_inv_dyn.pdf')

    # Plot 1 Schätzung inverse Dynamik
    double_subplot(
        samples_vec,
        [np.concatenate([tau_hat_test_Nruns_10[:, 0].reshape(-1, 1), tau_hat_test_Nruns_50[:, 0].reshape(-1, 1), labels_test[:, 0].reshape(-1, 1)], axis=1).reshape(-1, 3), 
        np.concatenate([tau_hat_test_Nruns_10[:, 1].reshape(-1, 1), tau_hat_test_Nruns_50[:, 1].reshape(-1, 1), labels_test[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 3)],
        r'Samples',
        [r'$F_{RS} \, / \, \mathrm{N}$', r'$\tau_{RS} \, / \, \mathrm{Nm}$'],
        ['', ''],
        [[r'DeLaN $N_r=10$', r'DeLaN $N_r=50$', 'GT'], [r'DeLaN $N_r=10$', r'DeLaN $N_r=50$', 'GT']],
        plot_1_name,
        False,
        True
    )

    # Ergebnisse speichern
    results = {
        'tau_hat_test_Nruns_10': tau_hat_test_Nruns_10,
        'rmse_tau_percent_Nruns_10': rmse_tau_percent_Nruns_10,
        'tau_hat_test_Nruns_50': tau_hat_test_Nruns_50,
        'rmse_tau_percent_Nruns_50': rmse_tau_percent_Nruns_50,
    }

    save_path = os.path.join(script_path, 'Ergebnisse_Variation_Datenpunkte.npy')

    np.save(save_path, results)

else:

    data_path = os.path.join(script_path, 'Ergebnisse_Variation_Datenpunkte.npy')

    # Ergebnis-Dict laden
    results = np.load(data_path, allow_pickle=True).item()

    # Ergebnisse entpacken
    tau_hat_test_Nruns_10 = results['tau_hat_test_Nruns_10']
    rmse_tau_percent_Nruns_10 = results['rmse_tau_percent_Nruns_10']
    tau_hat_test_Nruns_50 = results['tau_hat_test_Nruns_50']
    rmse_tau_percent_Nruns_50 = results['rmse_tau_percent_Nruns_50']

    # Prozentualen Fehler ausgeben
    print(f"Prozentualer Fehler Test Nr = 10: {rmse_tau_percent_Nruns_10:4f}")
    print(f"Prozentualer Fehler Test Nr = 50: {rmse_tau_percent_Nruns_50:4f}")

    # Ergebnisse plotten
    samples_vec = np.arange(1, tau_hat_test_Nruns_50.shape[0] + 1).reshape(-1, 1)

    plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
    plot_1_name = os.path.join(plot_path, 'Variation_Datenpunkte_inv_dyn.pdf')

    # Plot 1 Schätzung inverse Dynamik
    double_subplot(
        samples_vec,
        [np.concatenate([tau_hat_test_Nruns_10[:, 0].reshape(-1, 1), tau_hat_test_Nruns_50[:, 0].reshape(-1, 1), labels_test[:, 0].reshape(-1, 1)], axis=1).reshape(-1, 3), 
        np.concatenate([tau_hat_test_Nruns_10[:, 1].reshape(-1, 1), tau_hat_test_Nruns_50[:, 1].reshape(-1, 1), labels_test[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 3)],
        r'Samples',
        [r'$F_{RS} \, / \, \mathrm{N}$', r'$\tau_{RS} \, / \, \mathrm{Nm}$'],
        ['', ''],
        [[r'DeLaN $N_r=10$', r'DeLaN $N_r=50$', 'GT'], []],
        plot_1_name,
        True,
        True
    )