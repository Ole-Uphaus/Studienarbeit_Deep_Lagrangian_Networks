'''
Autor:      Ole Uphaus
Datum:     10.06.2025
Beschreibung:
In diesem Skript werde ich auf basis der Trajektoriendaten aus Matlab eine Folgeregelung (PD) erstellen. Dieses Skript dient dazu, den realen Anwendungsfall zu simulieren. Dabei werden zunächst solltrajektorien vorgegeben und diese so gut wie möglich mit dem Roboter abgefahren. Die dabei gesammelten Daten werden später zum Training des Modells verwendet. Ich werde lediglich die Trainingsdaten manipulieren. Die Tests werde ich später mit den in Matlab generierten Testdaten durchführen.
'''

import mujoco
from mujoco import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
from datetime import datetime

from DeLaN_functions_Ole import *

# Sollen Trainingsdaten erstellt werden
save_data = True

# Trainings- und Testdaten aus MATLAB File (es werden hier nur die Sollpositionen und Geschwindigkeiten benötigt)
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_06_07_09_09_04_Samples_3000.mat')  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_06_07_09_09_04_Samples_3000.mat')  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Daten aufbereiten (r, r_p und phi, phi_p heraussuchen) (zusätzlich noch Beschleunigungen - aber nur zur Auswertung)
r_des_traj = features_training[:, 0]
r_p_des_traj = features_training[:, 2]
r_pp_des_traj = features_training[:, 4]
phi_des_traj = features_training[:, 1]
phi_p_des_traj = features_training[:, 3]
phi_pp_des_traj = features_training[:, 5]

# Steuergrößen auswerten
tau_r_des_traj = labels_training[:, 1]
tau_phi_des_traj = labels_training[:, 0]

# Trajektorien aufteilen (da immer einzelne durchläufe in Datengenerierung - Dazu wird die Geschwindigkeit untersucht (Geschwindigkeit == 0 am Anfang und Ende))
threshold = 1.e-10  # Schwellenwert definieren

r_des_traj_divided = [] # Leere Arrays für einzelne Trajektorien
r_p_des_traj_divided = []
phi_des_traj_divided = []
phi_p_des_traj_divided = []

counter = 0 # Zähler
i_old = 0   # Vorheriger Index

for i in range(r_p_des_traj.shape[0]):
    if np.abs(r_p_des_traj[i]) < threshold:
        if np.mod(counter, 2) == 1:
            # Trajektorien anhängen
            r_des_traj_divided.append(r_des_traj[i_old:(i + 1)])
            r_p_des_traj_divided.append(r_p_des_traj[i_old:(i + 1)])
            phi_des_traj_divided.append(phi_des_traj[i_old:(i + 1)])
            phi_p_des_traj_divided.append(phi_p_des_traj[i_old:(i + 1)])
        
        # Zustände aktualisieren
        i_old = i
        counter += 1

# Listen in numpy arrays umwandeln
r_des_traj_divided = np.array(r_des_traj_divided)
r_p_des_traj_divided = np.array(r_p_des_traj_divided)
phi_des_traj_divided = np.array(phi_des_traj_divided)
phi_p_des_traj_divided = np.array(phi_p_des_traj_divided)

# Weitere Trajektoriendaten (Müssen mit Matlab Skript übereinstimmen)
move_time = 3
t_vec = np.linspace(0, move_time, r_des_traj_divided.shape[1])
dt = move_time/(r_des_traj_divided.shape[1] - 1)    # Zeitschritt hier etwas anders berechnet, da bei linspace ja immer ein Zeitschritt weniger als Anzahl der Samples

# XML Modell laden (Eigenes XML Modell nur für dieses Skript, da andere Abtastzeit)
xml_name = '2_FHG_Rob_Model_1_data_gen.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reglerparameter setzen
Kp = np.array([[50, 0],
               [0, 200]])

Kd = np.array([[50, 0],
               [0, 50]])

# Listen zum Tracken der Trajektorien
q_vec = []
q_p_vec = []
q_pp_vec = []

tau_vec = []

# Initiale Position einnehmen
data.qpos[0] = phi_des_traj_divided[0, 0]
data.qpos[1] = r_des_traj_divided[0, 0]

# Initiale Geschwindigkeiten setzen
data.qvel[0] = phi_p_des_traj_divided[0, 0]
data.qvel[1] = r_p_des_traj_divided[0, 0]

# Viewer starten
with viewer.launch_passive(model, data) as v:
    # Kamera konfigurieren
    v.cam.lookat[:] = [0.5, 0, 0]   # Zentrum deiner Szene
    v.cam.distance = 3.0                # Nähe
    v.cam.azimuth = 90
    v.cam.elevation = -70

    # Alle aufgeteilten Trajektorien durchgehen
    for i in range(r_des_traj_divided.shape[0]):

        # Initiale Position einnehmen
        data.qpos[0] = phi_des_traj_divided[i, 0]
        data.qpos[1] = r_des_traj_divided[i, 0]

        # Initiale Geschwindigkeiten setzen
        data.qvel[0] = phi_p_des_traj_divided[i, 0]
        data.qvel[1] = r_p_des_traj_divided[i, 0]

        # Initiale Position abspeichern
        q_vec.append(data.qpos.copy())
        q_p_vec.append(data.qvel.copy())

        # kurz warten am Anfang
        time.sleep(0.5)

        # Simulation beginnen
        for t in range(r_des_traj_divided.shape[1] - 1):   # Damit am Ende auch nur 100 Datenpunkte generiert werden
            # Startzeit
            loop_start = time.time()

            # Sollwerte auslesen
            phi_des = phi_des_traj_divided[i, t]
            r_des = r_des_traj_divided[i, t]

            phi_p_des = phi_p_des_traj_divided[i, t]
            r_p_des = r_p_des_traj_divided[i, t]

            # Ist-Werte auslesen
            phi = data.qpos[0]
            r = data.qpos[1]

            phi_p = data.qvel[0]
            r_p = data.qvel[1]

            # Fehler berechnen [e_phi; e_r]
            e = np.array([[phi_des - phi],
                        [r_des - r]])
            
            e_p = np.array([[phi_p_des - phi_p],
                        [r_p_des - r_p]])

            # PD-Regelung berechnen
            tau = np.matmul(Kp, e) + np.matmul(Kd, e_p)

            # Steuergrößen setzen
            data.ctrl[0] = tau[0, 0]
            data.ctrl[1] = tau[1, 0] 

            # Kinematik und Dynamik berechnen
            mujoco.mj_step(model, data)

            # Messwerte abspeichern
            # Istgelenkwinkel auslesen
            q_vec.append(data.qpos.copy())
            q_p_vec.append(data.qvel.copy())
            q_pp_vec.append(data.qacc.copy())

            # Steuergröße abspeichern
            tau_vec.append(np.array([tau[0, 0], tau[1, 0]]))

            # Bild aktualisieren
            v.sync()

            # Echtzeit synchronisieren
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Echtzeit überschritten bei t={t_vec[t]:.3f}s um {-sleep_time:.4f}s")

        # Noch einen weiteren step hinzufügen mit verschwindenden Stellgrößen, damit auch für den letzten datenpunkt eine Beschleunigung berechnet werden kann
        # Steuergrößen setzen
        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0

        # Kinematik und Dynamik berechnen
        mujoco.mj_step(model, data)

        # Gelenkbeschleunigung auslesen und abspeichern
        q_pp_vec.append(data.qacc.copy())

        # Steuergröße abspeichern
        tau_vec.append(np.array([0.0, 0.0]))

        # kurz warten am Ende
        time.sleep(0.5)

# Messwerte in np array umwandeln
q_vec = np.array(q_vec)
q_p_vec = np.array(q_p_vec)
q_pp_vec = np.array(q_pp_vec)

tau_vec = np.array(tau_vec)

# Daten bei Bedarf abspeichern
if save_data:
    # Zusammensetzen der Vektoren (Damit alles identisch wie in Matlab Simulation)
    features_training_PD = np.concatenate([q_vec[:, 1].reshape(-1, 1), q_vec[:, 0].reshape(-1, 1),
                                           q_p_vec[:, 1].reshape(-1, 1), q_p_vec[:, 0].reshape(-1, 1), 
                                           tau_vec[:, 1].reshape(-1, 1), tau_vec[:, 0].reshape(-1, 1)], axis=1) # Reihenfolge von r und phi vertauschen
    labels_training_PD = np.concatenate([q_pp_vec[:, 1].reshape(-1, 1), q_pp_vec[:, 0].reshape(-1, 1)], axis=1)    # Reihenfolge von r und phi vertauschen

    # Vektoren nun wieder so zusammensetzen wie ursprünglich in matlab
    features_test_PD = np.concatenate([features_test[:, :4], labels_test], axis=1)
    labels_test_PD = features_test[:, 4:]

    # Dictionary erstellen
    save_dict = {
        'features_training': features_training_PD,
        'labels_training': labels_training_PD,
        'features_test': features_test_PD,
        'labels_test': labels_test_PD,
        'Mass_Cor_test': Mass_Cor_test
    }

    # Pfad zum Speichern erstellen
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = os.path.join(script_path, '..', '..', 'Training_Data', 'Mujoco_Simulation', f"SimData_Mujoco_PD_{timestamp}.mat")

    # .mat file erstellen
    savemat(save_path, save_dict)

# Gesamtzeitvektor erstellen
t_vec_ges = np.linspace(0, int(move_time*counter/2), int(r_des_traj_divided.shape[1]*counter/2))

# Soll vs. Ist Gelenkkoordinaten Plotten
plt.figure()

plt.subplot(2, 3, 1)
plt.plot(t_vec_ges, phi_des_traj, label='phi soll')
plt.plot(t_vec_ges, q_vec[:, 0], label='phi ist')
plt.title('phi')
plt.xlabel('t')
plt.ylabel('phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t_vec_ges, r_des_traj, label='r soll')
plt.plot(t_vec_ges, q_vec[:, 1], label='r ist')
plt.title('r')
plt.xlabel('t')
plt.ylabel('r')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t_vec_ges, phi_p_des_traj, label='phip soll')
plt.plot(t_vec_ges, q_p_vec[:, 0], label='phip ist')
plt.title('phip')
plt.xlabel('t')
plt.ylabel('phip')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t_vec_ges, r_p_des_traj, label='rp soll')
plt.plot(t_vec_ges, q_p_vec[:, 1], label='rp ist')
plt.title('rp')
plt.xlabel('t')
plt.ylabel('rp')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t_vec_ges, phi_pp_des_traj, label='phipp soll')
plt.plot(t_vec_ges, q_pp_vec[:, 0], label='phipp ist')
plt.title('phipp')
plt.xlabel('t')
plt.ylabel('phipp')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t_vec_ges, r_pp_des_traj, label='rpp soll')
plt.plot(t_vec_ges, q_pp_vec[:, 1], label='rpp ist')
plt.title('rpp')
plt.xlabel('t')
plt.ylabel('rpp')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t_vec_ges, tau_phi_des_traj, label='tauphi soll')
plt.plot(t_vec_ges, tau_vec[:, 0], label='tauphi ist')
plt.title('tauphi')
plt.xlabel('t')
plt.ylabel('tauphi')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_vec_ges, tau_r_des_traj, label='taur soll')
plt.plot(t_vec_ges, tau_vec[:, 1], label='taur ist')
plt.title('taur')
plt.xlabel('t')
plt.ylabel('taur')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()
