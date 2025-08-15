'''
Autor:      Ole Uphaus
Datum:     29.07.2025
Beschreibung:
In diesem Skript wird das DeLaN modell auf dem standard Datensatz Trainiert. Dabei werden die internen Aktivierungsfunktionen variiert, um deren Einfluss zu zeigen.
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

###################### Konfiguration ELU ######################

# Parameter festlegen
hyper_param = {
    # Netzparameter
    'hidden_width': 64,
    'hidden_depth': 2,
    'activation_fnc': 'relu',
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
    'use_inverse_model': True,
    'use_forward_model': True,
    'use_energy_consumption': False,
    'save_model': False}

# Trainings- und Testdaten laden
target_folder = 'Studienarbeit_Data' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen' 'Studienarbeit_Data'
features_training, labels_training, _, _, _ = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('Allgemeiner_Trainingsdatensatz_Nruns_37.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

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
training_loss_history_ReLU = np.array(results['training_loss_history'])
test_loss_history_ReLU = np.array(results['test_loss_history'])

###################### Konfiguration Softplus ######################
hyper_param['activation_fnc'] = 'softplus'

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
training_loss_history_Softplus = np.array(results['training_loss_history'])
test_loss_history_Softplus = np.array(results['test_loss_history'])

###################### Konfiguration ReLU ######################
hyper_param['activation_fnc'] = 'elu'

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
training_loss_history_ELU = np.array(results['training_loss_history'])
test_loss_history_ELU = np.array(results['test_loss_history'])

# Ergebnisse plotten
plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
plot_1_name = os.path.join(plot_path, 'Opt_Hyper_Aktivierungsfunktion_Intern.pdf')

# Plot 2 Loss Entwicklung
single_plot_log(
    test_loss_history_ELU[:, 0],
    np.concatenate([test_loss_history_ELU[:, 1].reshape(-1, 1), test_loss_history_Softplus[:, 1].reshape(-1, 1), test_loss_history_ReLU[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 3),
    r'Epochen',
    r'$\mathrm{Test-Loss}$',
    '',
    ['ELU', 'Softplus', 'ReLU'],
    plot_1_name,
    True,
    True
)
