'''
Autor:      Ole Uphaus
Datum:     29.07.2025
Beschreibung:
In diesem Skript wird das DeLaN modell auf dem standard Datensatz Trainiert. Dabei werden die Verlustfunktion variiert, um deren Einfluss zu zeigen.
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

# Auswählen, ob nur geplottet oder auch trainiert werden soll
just_plot = True

if just_plot == False:
    ###################### Konfiguration vor. dyn. ######################

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
        'n_epoch': 2000,

        # Reibungsmodell
        'use_friction_model': False,
        'friction_model_init_d': [0.01, 0.01],
        'friction_model_init_c': [0.01, 0.01],
        'friction_model_init_s': [0.01, 0.01],
        'friction_model_init_v': [0.01, 0.01],
        'friction_epsilon': 100.0,

        # Sonstiges
        'use_inverse_model': False,
        'use_forward_model': True,
        'use_energy_consumption': False,
        'save_model': False}

    # Trainings- und Testdaten laden
    target_folder = 'Studienarbeit_Data' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen' 'Studienarbeit_Data'
    features_training, labels_training, _, _, _ = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Trainingsdaten
    _, _, features_test, labels_test, Mass_Cor_test = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

    print('\n Konfiguration vor.')

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
    training_loss_history_Vor = np.array(results['training_loss_history'])
    test_loss_history_Vor = np.array(results['test_loss_history'])
    mae_tau_percent_Vor= results['mae_tau_percent']

    ###################### Konfiguration energy cons ######################
    hyper_param['use_inverse_model'] = False
    hyper_param['use_forward_model'] = False
    hyper_param['use_energy_consumption'] = True

    print('\n Konfiguration Ener.')

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
    training_loss_history_Ener = np.array(results['training_loss_history'])
    test_loss_history_Ener = np.array(results['test_loss_history'])
    mae_tau_percent_Ener= results['mae_tau_percent']

    ###################### Konfiguration inv + vor ######################
    hyper_param['use_inverse_model'] = True
    hyper_param['use_forward_model'] = True
    hyper_param['use_energy_consumption'] = False

    print('\n Konfiguration inv. + vor.')

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
    training_loss_history_inv_Vor = np.array(results['training_loss_history'])
    test_loss_history_inv_Vor = np.array(results['test_loss_history'])
    mae_tau_percent_inv_Vor= results['mae_tau_percent']

    ###################### Konfiguration inv ######################
    hyper_param['use_inverse_model'] = True
    hyper_param['use_forward_model'] = False
    hyper_param['use_energy_consumption'] = False

    print('\n Konfiguration inv.')

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
    training_loss_history_inv = np.array(results['training_loss_history'])
    test_loss_history_inv = np.array(results['test_loss_history'])
    mae_tau_percent_inv= results['mae_tau_percent']

    ###################### Konfiguration inv + vor + energy ######################
    hyper_param['use_inverse_model'] = True
    hyper_param['use_forward_model'] = True
    hyper_param['use_energy_consumption'] = True

    print('\n Konfiguration inv. + vor. + Ener.')

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
    training_loss_history_inv_Vor_Ener = np.array(results['training_loss_history'])
    test_loss_history_inv_Vor_Ener = np.array(results['test_loss_history'])
    mae_tau_percent_inv_Vor_Ener= results['mae_tau_percent']

    ###################### Konfiguration vor + energy ######################
    hyper_param['use_inverse_model'] = False
    hyper_param['use_forward_model'] = True
    hyper_param['use_energy_consumption'] = True

    print('\n Konfiguration vor. + ener.')

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
    training_loss_history_Vor_Ener = np.array(results['training_loss_history'])
    test_loss_history_Vor_Ener = np.array(results['test_loss_history'])
    mae_tau_percent_Vor_Ener= results['mae_tau_percent']

    # Ergebnisse speichern
    results = {
        'training_loss_history_Vor': training_loss_history_Vor,
        'test_loss_history_Vor': test_loss_history_Vor,
        'mae_tau_percent_Vor': mae_tau_percent_Vor,

        'training_loss_history_Ener': training_loss_history_Ener,
        'test_loss_history_Ener': test_loss_history_Ener,
        'mae_tau_percent_Ener': mae_tau_percent_Ener,

        'training_loss_history_inv_Vor': training_loss_history_inv_Vor,
        'test_loss_history_inv_Vor': test_loss_history_inv_Vor,
        'mae_tau_percent_inv_Vor': mae_tau_percent_inv_Vor,

        'training_loss_history_inv': training_loss_history_inv,
        'test_loss_history_inv': test_loss_history_inv,
        'mae_tau_percent_inv': mae_tau_percent_inv,

        'training_loss_history_inv_Vor_Ener': training_loss_history_inv_Vor_Ener,
        'test_loss_history_inv_Vor_Ener': test_loss_history_inv_Vor_Ener,
        'mae_tau_percent_inv_Vor_Ener': mae_tau_percent_inv_Vor_Ener,

        'training_loss_history_Vor_Ener': training_loss_history_Vor_Ener,
        'test_loss_history_Vor_Ener': test_loss_history_Vor_Ener,
        'mae_tau_percent_Vor_Ener': mae_tau_percent_Vor_Ener
    }

    save_path = os.path.join(script_path, 'Ergebnisse_Verlustfkt.npy')

    np.save(save_path, results)

    # Ergebnisse plotten
    plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
    plot_1_name = os.path.join(plot_path, 'Opt_Hyper_Verlustfkt.pdf')

    # Plot 2 Loss Entwicklung
    single_plot_log(
        test_loss_history_Vor_Ener[:, 0],
        np.concatenate([test_loss_history_Vor[:, 1].reshape(-1, 1), test_loss_history_inv[:, 1].reshape(-1, 1), test_loss_history_Ener[:, 1].reshape(-1, 1), test_loss_history_inv_Vor[:, 1].reshape(-1, 1), test_loss_history_Vor_Ener[:, 1].reshape(-1, 1), test_loss_history_inv_Vor_Ener[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 6),
        r'Epochen',
        r'$\mathrm{Test-Loss}$',
        '',
        ['Vor.', 'inv.', 'Ener.', 'inv. + Vor.', 'Vor. + Ener.', 'inv. + Vor. + Ener.'],
        plot_1_name,
        False,
        True
    )

else:

    data_path = os.path.join(script_path, 'Ergebnisse_Verlustfkt.npy')

    # Ergebnis-Dict laden
    results = np.load(data_path, allow_pickle=True).item()

    # Ergebnisse entpacken
    training_loss_history_Vor = results['training_loss_history_Vor']
    test_loss_history_Vor = results['test_loss_history_Vor']
    mae_tau_percent_Vor = results['mae_tau_percent_Vor']

    training_loss_history_Ener = results['training_loss_history_Ener']
    test_loss_history_Ener = results['test_loss_history_Ener']
    mae_tau_percent_Ener = results['mae_tau_percent_Ener']

    training_loss_history_inv_Vor = results['training_loss_history_inv_Vor']
    test_loss_history_inv_Vor = results['test_loss_history_inv_Vor']
    mae_tau_percent_inv_Vor = results['mae_tau_percent_inv_Vor']

    training_loss_history_inv = results['training_loss_history_inv']
    test_loss_history_inv = results['test_loss_history_inv']
    mae_tau_percent_inv = results['mae_tau_percent_inv']

    training_loss_history_inv_Vor_Ener = results['training_loss_history_inv_Vor_Ener']
    test_loss_history_inv_Vor_Ener = results['test_loss_history_inv_Vor_Ener']
    mae_tau_percent_inv_Vor_Ener = results['mae_tau_percent_inv_Vor_Ener']

    training_loss_history_Vor_Ener = results['training_loss_history_Vor_Ener']
    test_loss_history_Vor_Ener = results['test_loss_history_Vor_Ener']
    mae_tau_percent_Vor_Ener = results['mae_tau_percent_Vor_Ener']

    # Prozentualen Fehler ausgeben
    print(f"Prozentualer MAE Test Vor: {mae_tau_percent_Vor:4f}")
    print(f"Prozentualer MAE Test Ener: {mae_tau_percent_Ener:4f}")
    print(f"Prozentualer MAE Test inv: {mae_tau_percent_inv:4f}")
    print(f"Prozentualer MAE Test inv + Vor: {mae_tau_percent_inv_Vor:4f}")
    print(f"Prozentualer MAE Test inv + Vor + Ener: {mae_tau_percent_inv_Vor_Ener:4f}")
    print(f"Prozentualer MAE Test Vor + Ener: {mae_tau_percent_Vor_Ener:4f}")

    # Ergebnisse plotten
    plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
    plot_1_name = os.path.join(plot_path, 'Opt_Hyper_Verlustfkt.pdf')

    # Plot 2 Loss Entwicklung
    single_plot_log(
        test_loss_history_Vor_Ener[:, 0],
        np.concatenate([test_loss_history_Vor[:, 1].reshape(-1, 1), test_loss_history_inv[:, 1].reshape(-1, 1), test_loss_history_Ener[:, 1].reshape(-1, 1), test_loss_history_inv_Vor[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 4),
        r'Epochen',
        r'$\mathrm{Verlust}$',
        '',
        [r'$\ell_{Vor}(\cdot)$', r'$\ell_{inv}(\cdot)$', r'$\ell_{Ener}(\cdot)$', r'$\ell_{inv}(\cdot) + \ell_{Vor}(\cdot)$'],
        plot_1_name,
        True,
        True
    )
