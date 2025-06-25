'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
In diesem Skript wird das von mir erstellte Deep Lagrangien Network trainiert. Hier ist Raum für allgemeine Tests und Erprobungen der Hyperparameter.
'''

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

from DeLaN_model_Ole import Deep_Lagrangian_Network
from DeLaN_functions_Ole import *

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
    'L_diagonal_offset': 1.e-3,
    
    # Training
    'dropuot': 0.0,
    'batch_size': 512,
    'learning_rate': 5.e-4,
    'weight_decay': 1.e-4,
    'n_epoch': 500,

    # Reibungsmodell
    'use_friction_model': False,
    'friction_model_init_d': [0.01, 0.01],
    'friction_model_init_c': [0.01, 0.01],
    'friction_model_init_s': [0.01, 0.01],
    'friction_model_init_v': [0.01, 0.01],
    'friction_epsilon': 100.0,

    # Sonstiges
    'use_inverse_model': True,
    'use_forward_model': False,
    'save_model': False}

# Trainings- und Testdaten laden
target_folder = 'MATLAB_Simulation' # Möglichkeiten: 'MATLAB_Simulation', 'Mujoco_Simulation', 'Torsionsschwinger_Messungen'
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat', target_folder)  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Torch Tensoren der Trainingsdaten erstellen
features_training_tensor = torch.tensor(features_training, dtype=torch.float32)
labels_training_tensor = torch.tensor(labels_training, dtype=torch.float32)

# Dataset und Dataloader für das Training erstellen
dataset_training = TensorDataset(features_training_tensor, labels_training_tensor)
dataloader_training = DataLoader(dataset_training, batch_size=hyper_param['batch_size'], shuffle=True, drop_last=True)

# Testdaten in torch Tensoren umwandeln
features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

# Testdaten zuordnen und auf device verschieben
q_test = features_test_tensor[:, (0, 1)].to(device)
qd_test = features_test_tensor[:, (2, 3)].to(device)
qdd_test = features_test_tensor[:, (4, 5)].to(device)
tau_test = labels_test_tensor.to(device)
tau_test_plot = labels_test_tensor.cpu().numpy()    # Diesen Tensor direkt auf cpu schieben, damit damit nachher der loss berechnet werden kann

# Ausgabe Datendimensionen
print('Datenpunkte Training: ', features_training.shape[0])
print('Datenpunkte Evaluierung: ', features_test.shape[0])
print()

# DeLaN Netzwerk erstellen
n_dof = labels_training.shape[1]
DeLaN_network = Deep_Lagrangian_Network(n_dof, **hyper_param).to(device)

# Optimierer Initialisieren
optimizer = torch.optim.Adam(DeLaN_network.parameters(),
                                lr=hyper_param["learning_rate"],
                                weight_decay=hyper_param["weight_decay"],
                                amsgrad=True)

# Optimierung starten und Zeitmessung beginnen
print('Starte Optimierung...')
print()
start_time = time.time()

# Training des Netzwerks
training_loss_history = []
test_loss_history = []
output_L_diag_no_activation_history = []

for epoch in range(hyper_param['n_epoch']):
    # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
    DeLaN_network.train()
    loss_sum = 0
    output_L_diag_no_activation_mean_sum = 0

    for batch_features, batch_labels in dataloader_training:
        # Gradienten zurücksetzen
        optimizer.zero_grad()

        # Trainingsdaten zuordnen
        q = batch_features[:, (0, 1)].to(device)
        qd = batch_features[:, (2, 3)].to(device)
        qdd = batch_features[:, (4, 5)].to(device)
        tau = batch_labels.to(device)

        # Loss initialisieren
        loss = torch.tensor(0.0, device=device)

        if hyper_param['use_inverse_model']:
            # Forward pass
            tau_hat, _, _, _, _, output_L_diag_no_activation = DeLaN_network(q, qd, qdd)    # Inverses Modell

            # Durchnittlichen wert der Diagonalelemente vor der ReLu aktivierung (über Batch gemittelt)
            output_L_diag_no_activation_mean = output_L_diag_no_activation.mean(dim=0)

            # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
            err_inv_dyn = torch.sum((tau_hat - tau)**2, dim=1)
            mean_err_inv_dyn = torch.mean(err_inv_dyn)

            # Loss berechnen
            loss += mean_err_inv_dyn

        if hyper_param['use_forward_model']:
            # Forward pass    
            qdd_hat, _, _, _ = DeLaN_network.forward_dynamics(q, qd, tau) # Vorwärts Modell

            # Fehler aus Vorwärtsmodell berechnen (Schätzung von qdd)
            err_for_dyn = torch.sum((qdd_hat - qdd)**2, dim=1)
            mean_err_for_dyn = torch.mean(err_for_dyn)

            # Loss berechnen
            loss += mean_err_for_dyn

        if hyper_param['use_inverse_model'] == False and hyper_param['use_forward_model'] == False:
            raise ValueError("Ungültige Konfiguration: 'use_inverse_model' und 'use_forward_model' dürfen nicht beide False sein.")

        # Optimierungsschritt durchführen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(DeLaN_network.parameters(), max_norm=0.5)    # Gradienten Clipping für besseres Training
        optimizer.step()

        # Loss des aktuellen Batches aufsummieren (und mittleren Wert der Diagonalelemente ohne aktivierung)
        loss_sum += loss.item()
        if hyper_param['use_inverse_model']:
            output_L_diag_no_activation_mean_sum += output_L_diag_no_activation_mean

    # Mittleren Loss berechnen und ausgeben (und mittleren Wert der Diagonalelemente ohne aktivierung)
    loss_mean_batch = loss_sum/len(dataloader_training)
    if hyper_param['use_inverse_model']:
        output_L_diag_no_activation_mean_batch = output_L_diag_no_activation_mean_sum/len(dataloader_training)

    # Loss an Loss history anhängen (und mittleren Wert der Diagonalelemente ohne aktivierung)
    training_loss_history.append([epoch + 1, loss_mean_batch])
    if hyper_param['use_inverse_model']:
        output_L_diag_no_activation_history.append(output_L_diag_no_activation_mean_batch.cpu().detach().numpy())

    if epoch == 0 or np.mod(epoch + 1, 100) == 0:
        # Model Evaluieren
        test_loss, _, _, _, _, _ = model_evaluation(DeLaN_network, q_test, qd_test, qdd_test, tau_test, hyper_param['use_inverse_model'], hyper_param['use_forward_model'])

        # Loss an Loss history anhängen
        test_loss_history.append([epoch + 1, test_loss])

        # Ausgabe während des Trainings
        print(f'Epoch [{epoch + 1}/{hyper_param['n_epoch']}], Training-Loss: {loss_mean_batch:.3e}, Test-Loss: {test_loss:.3e}, Verstrichene Zeit: {(time.time() - start_time):.2f} s')

# Modell evaluieren (kein torch.nograd(), da interne Gradienten benötigt werden)
DeLaN_network.eval()

# Evaluierung
_, tau_hat_test, H_test, c_test, g_test, tau_fric_test = model_evaluation(DeLaN_network, q_test, qd_test, qdd_test, tau_test, hyper_param['use_inverse_model'], hyper_param['use_forward_model'])

# Metriken Berechnen (Fehler aus inverser Dynamik)
mse_tau = np.mean(np.sum((tau_hat_test - tau_test.cpu().detach().numpy())**2, axis=1))
rmse_tau = np.sqrt(mse_tau)
tau_mean = np.sqrt(np.mean(np.sum((tau_test.cpu().detach().numpy())**2, axis=1)))
rmse_tau_percent = rmse_tau/tau_mean*100

# Metriken ausgeben
print('Metriken:')
print(f"MSE Test: {mse_tau:4f}")
print(f"RMSE (Absolutfehler) Test: {rmse_tau:4f}")
print(f"Prozentualer Fehler Test: {rmse_tau_percent:4f}")

# Modell abspeichern
if hyper_param['save_model'] == True:
        
    # Aktueller Zeitstempel
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    script_path = os.path.dirname(os.path.abspath(__file__))

    if target_folder == 'MATLAB_Simulation':
        model_path = os.path.join(script_path, "Saved_Models", f"DeLaN_model_{timestamp}_Epochen_{hyper_param['n_epoch']}.pth")
    else:
        model_path = os.path.join(script_path, "Saved_Models", f"DeLaN_model_MJ_Sim_{timestamp}_Epochen_{hyper_param['n_epoch']}.pth")

    # Speichern
    torch.save({
        'state_dict': DeLaN_network.state_dict(),
        'hyper_param': hyper_param,
        'n_dof': n_dof
    }, model_path)

# Wenn Reibungsmodell gewählt, dann Reibungsparameter ausgeben
if hyper_param['use_friction_model']:
    print('Reibungsparameter:')
    print(f"Dämpfung (viskos): {DeLaN_network.friction_d().detach().cpu().numpy().tolist()}")
    print(f"Coulomb-Reibung: {DeLaN_network.friction_c().detach().cpu().numpy().tolist()}")
    print(f"Stribeck-Spitze: {DeLaN_network.friction_s().detach().cpu().numpy().tolist()}")
    print(f"Stribeck-Breite: {DeLaN_network.friction_v().detach().cpu().numpy().tolist()}")

# Plotten
samples_vec = np.arange(1, H_test.shape[0] + 1)

# Loss Entwicklung plotten (und mittleren Wert der Diagonalelemente ohne aktivierung)
training_loss_history = np.array(training_loss_history)
test_loss_history = np.array(test_loss_history)
output_L_diag_no_activation_history = np.array(output_L_diag_no_activation_history)

plt.figure()

plt.subplot(2, 1, 1)
plt.semilogy(training_loss_history[:, 0], training_loss_history[:, 1], label='Training Loss')
plt.semilogy(test_loss_history[:, 0], test_loss_history[:, 1], label='Test Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.title('Loss-Verlauf während des Trainings')
plt.grid(True)
plt.legend()

if hyper_param['use_inverse_model']:
    plt.subplot(2, 1, 2)
    plt.plot(training_loss_history[:, 0], output_L_diag_no_activation_history[:, 0], label='H11')
    plt.plot(training_loss_history[:, 0], output_L_diag_no_activation_history[:, 1], label='H22')
    plt.title('H ohne Aktivierung')
    plt.xlabel('Epoche')
    plt.ylabel('H')
    plt.grid(True)
    plt.legend()

plt.tight_layout()

# H
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(samples_vec, H_test[:, 0, 0], label='H11 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 0] ,label='H11 Analytic')
plt.title('H11')
plt.xlabel('Samples')
plt.ylabel('H')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(samples_vec, H_test[:, 0, 1], label='H12 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 1], label='H12 Analytic')
plt.title('H12')
plt.xlabel('Samples')
plt.ylabel('H')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(samples_vec, H_test[:, 1, 0], label='H21 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 1], label='H21 Analytic')
plt.title('H21')
plt.xlabel('Samples')
plt.ylabel('H')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(samples_vec, H_test[:, 1, 1], label='H22 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 2], label='H22 Analytic')
plt.title('H22')
plt.xlabel('Samples')
plt.ylabel('H')
plt.grid(True)
plt.legend()

plt.tight_layout()

# c
plt.figure()

plt.subplot(2, 3, 1)
plt.plot(samples_vec, c_test[:, 0], label='C1 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 3] ,label='C1 Analytic')
plt.title('C1')
plt.xlabel('Samples')
plt.ylabel('C')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(samples_vec, c_test[:, 1], label='C2 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 4] ,label='C2 Analytic')
plt.title('C2')
plt.xlabel('Samples')
plt.ylabel('C')
plt.grid(True)
plt.legend()

# g
plt.subplot(2, 3, 2)
plt.plot(samples_vec, g_test[:, 0], label='g1 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 5] ,label='g1 Analytic')
plt.title('g1')
plt.xlabel('Samples')
plt.ylabel('g')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(samples_vec, g_test[:, 1], label='g2 DeLaN')
plt.plot(samples_vec, Mass_Cor_test[:, 6] ,label='g2 Analytic')
plt.title('g2')
plt.xlabel('Samples')
plt.ylabel('g')
plt.grid(True)
plt.legend()

# tau
plt.subplot(2, 3, 3)
plt.plot(samples_vec, tau_hat_test[:, 0], label='tau1 DeLaN')
plt.plot(samples_vec, tau_test_plot[:, 0] ,label='tau1 Analytic')
plt.title('tau1')
plt.xlabel('Samples')
plt.ylabel('tau')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(samples_vec, tau_hat_test[:, 1], label='tau2 DeLaN')
plt.plot(samples_vec, tau_test_plot[:, 1] ,label='tau2 Analytic')
plt.title('tau2')
plt.xlabel('Samples')
plt.ylabel('tau')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Reibungskräfte
if hyper_param['use_friction_model'] and Mass_Cor_test.shape[1] > 8:
    # Reibungskennlinie auswerten
    qd_numpy, tau_fric_numpy = eval_friction_graph(DeLaN_network, device)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(samples_vec, tau_fric_test[:, 0], label='fric1 DeLaN')
    plt.plot(samples_vec, Mass_Cor_test[:, 7] ,label='fric1 Analytic')
    plt.title('fric1')
    plt.xlabel('Samples')
    plt.ylabel('fric1')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(samples_vec, tau_fric_test[:, 1], label='fric2 DeLaN')
    plt.plot(samples_vec, Mass_Cor_test[:, 8], label='fric2 Analytic')
    plt.title('fric2')
    plt.xlabel('Samples')
    plt.ylabel('fric2')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(qd_numpy, tau_fric_numpy[:, 0] ,label='fric1')
    plt.title('Reibungskennlinie fric1')
    plt.xlabel('qd')
    plt.ylabel('fric1')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(qd_numpy, tau_fric_numpy[:, 1] ,label='fric2')
    plt.title('Reibungskennlinie fric2')
    plt.xlabel('qd')
    plt.ylabel('fric2')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

plt.show()