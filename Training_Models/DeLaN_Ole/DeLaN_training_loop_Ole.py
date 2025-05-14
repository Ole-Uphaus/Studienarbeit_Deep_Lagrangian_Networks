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

from DeLaN_model_Ole import Deep_Lagrangian_Network
from DeLaN_functions_Ole import *

# Checken, ob Cuda verfügbar und festlegen des devices, auf dem trainiert werden soll
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benutze Device: {device}")
print()

# Seeds ausprobieren
max_seed = 100
seed_vec = np.arange(1, max_seed + 1)

# Ergebnisvektor
seed_loss_vec = np.zeros([max_seed, 2])
seed_loss_vec[:, 0] = seed_vec

# Loop, um mehrere Seeds auszuprobieren
for i_seed in seed_vec:

    print('Aktueller Seed: ', i_seed)

    # Seed setzen für Reproduzierbarkeit
    seed = i_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Parameter festlegen
    hyper_param = {
        # Netzparameter
        'hidden_width': 32,
        'hidden_depth': 3,
        'activation_fnc': 'elu',

        # Initialisierung
        'bias_init_constant': 1.e-3,
        'wheight_init': 'xavier_normal',

        # Lagrange Dynamik
        'L_diagonal_offset': 1.e-4,
        
        # Training
        'dropuot': 0.0,
        'batch_size': 512,
        'learning_rate': 5.e-4,
        'weight_decay': 1.e-4,
        'n_epoch': 2000,

        # Sonstiges
        'save_model': False}

    # Trainings- und Testdaten laden 
    features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat')  # Mein Modell Trainingsdaten
    _, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_27_03_Samples_3000.mat')  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

    # Torch Tensoren der Trainingsdaten erstellen
    features_training_tensor = torch.tensor(features_training, dtype=torch.float32)
    labels_training_tensor = torch.tensor(labels_training, dtype=torch.float32)

    # Dataset und Dataloader für das Training erstellen
    dataset_training = TensorDataset(features_training_tensor, labels_training_tensor)
    dataloader_training = DataLoader(dataset_training, batch_size=hyper_param['batch_size'], shuffle=True, drop_last=True)

    # Testdaten in torch Tensoren umwandeln
    features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

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
    loss_history = []
    for epoch in range(hyper_param['n_epoch']):
        # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
        DeLaN_network.train()
        loss_sum = 0

        for batch_features, batch_labels in dataloader_training:
            # Gradienten zurücksetzen
            optimizer.zero_grad()

            # Trainingsdaten zuordnen
            q = batch_features[:, (0, 1)].to(device)
            qd = batch_features[:, (2, 3)].to(device)
            qdd = batch_features[:, (4, 5)].to(device)
            tau = batch_labels.to(device)

            # Forward pass
            tau_hat, _, _, _ = DeLaN_network(q, qd, qdd)

            # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
            err_inv_dyn = torch.sum((tau_hat - tau)**2, dim=1)
            mean_err_inv_dyn = torch.mean(err_inv_dyn)

            # Loss berechnen und Optimierungsschritt durchführen
            loss = mean_err_inv_dyn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DeLaN_network.parameters(), max_norm=0.5)    # Gradienten Clopping für besseres Training
            optimizer.step()

            # Loss des aktuellen Batches aufsummieren
            loss_sum += loss.item()

        # Mittleren Loss berechnen und ausgeben
        loss_mean_batch = loss_sum/len(dataloader_training)

        # Loss an Loss history anhängen
        loss_history.append([epoch + 1, loss_mean_batch])

        if epoch == 0 or np.mod(epoch + 1, 100) == 0:
            # Ausgabe während des Trainings
            print(f'Epoch [{epoch + 1}/{hyper_param['n_epoch']}], Training-Loss: {loss_mean_batch:.3e}, Verstrichene Zeit: {(time.time() - start_time):.2f} s')

    # Modell evaluieren (kein torch.nograd(), da interne Gradienten benötigt werden)
    DeLaN_network.eval()

    # Testdaten zuordnen und auf device verschieben
    q_test = features_test_tensor[:, (0, 1)].to(device)
    qd_test = features_test_tensor[:, (2, 3)].to(device)
    qdd_test = features_test_tensor[:, (4, 5)].to(device)
    tau_test = labels_test_tensor.cpu().numpy()    # Diesen Tensor direkt auf cpu schieben, damit damit nachher der loss berechnet werden kann

    # Prädiktion
    print()
    print('Evaluierung...')

    # Forward pass
    out_test = DeLaN_network(q_test, qd_test, qdd_test)

    tau_hat_test = out_test[0].cpu().detach().numpy()   # Tesnoren auf cpu legen, gradienten entfernen, un numpy arrays umwandeln

    # Test loss berechnen (um mit Training zu vergleichen)
    err_inv_dyn_test = np.sum((tau_hat_test - tau_test)**2, axis=1)
    mean_err_inv_dyn_test = np.mean(err_inv_dyn_test)

    # Ausgabe loss
    print(f'Test-Loss: {mean_err_inv_dyn_test:.3e}')
    print()

    # Fehler abspeichern
    seed_loss_vec[i_seed - 1, 1] = mean_err_inv_dyn_test

print(seed_loss_vec)

# Fehler Visualisieren
plt.figure()
plt.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 1])
plt.xscale("linear")
plt.yscale("log") 
plt.xlabel("Seed")
plt.ylabel("Fehler MSE")
plt.title("Zusammenhang zwischen Seed und Fehler")
plt.grid(True)
plt.show()