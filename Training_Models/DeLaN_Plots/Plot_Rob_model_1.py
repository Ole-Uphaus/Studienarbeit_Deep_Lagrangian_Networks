'''
Autor:      Ole Uphaus
Datum:     29.07.2025
Beschreibung:
Dies ist ein plot skript für das DeLaN Modell.
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

# Parameter festlegen
hyper_param = {
    # Netzparameter
    'hidden_width': 64,
    'hidden_depth': 2,
    'activation_fnc': 'elu',
    'activation_fnc_diag': 'relu',

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
    'n_epoch': 500,

    # Reibungsmodell
    'use_friction_model': False,
    'friction_model_init_d': [0.01, 0.01],
    'friction_model_init_c': [3.01, 0.01],
    'friction_model_init_s': [2.01, 0.01],
    'friction_model_init_v': [0.01, 0.01],
    'friction_epsilon': 100.0,

    # Sonstiges
    'use_inverse_model': True,
    'use_forward_model': True,
    'use_energy_consumption': False,
    'save_model': False}

# Trainings- und Testdaten laden
target_folder = 'MATLAB_Simulation' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen'
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

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
training_loss_history = results['training_loss_history']
test_loss_history = results['test_loss_history']
output_L_diag_no_activation_history = results['output_L_diag_no_activation_history']
H_test = results['H_test']
c_test = results['c_test']
g_test = results['g_test']
tau_fric_test = results['tau_fric_test']
tau_hat_test = results['tau_hat_test']

# Ergebnisse plotten
samples_vec = np.arange(1, H_test.shape[0] + 1).reshape(-1, 1)

plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
plot_1_name = os.path.join(plot_path, 'test_plot.pdf')

# Plot Schätzung inverse Dynamik
double_subplot(
    samples_vec,
    [np.concatenate([tau_hat_test[:, 0].reshape(-1, 1), labels_test[:, 0].reshape(-1, 1)], axis=1).reshape(-1, 2), 
     np.concatenate([tau_hat_test[:, 1].reshape(-1, 1), labels_test[:, 1].reshape(-1, 1)], axis=1).reshape(-1, 2)],
    r'Samples',
    [r'$F_{RS} \, / \, \mathrm{N}$', r'$\tau_{RS} \, / \, \mathrm{Nm}$'],
    ['', ''],
    [['DeLaN', 'GT'], ['DeLaN', 'GT']],
    plot_1_name,
    True,
    True
)