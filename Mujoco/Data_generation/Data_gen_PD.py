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

from DeLaN_functions_Ole import *

# Trainings- und Testdaten aus MATLAB File (es werden hier nur die Sollpositionen und Geschwindigkeiten benötigt)
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_06_07_09_09_04_Samples_3000.mat')  # Mein Modell Trainingsdaten
_, _, features_test, labels_test, Mass_Cor_test = extract_training_data('SimData_V3_Rob_Model_1_2025_06_07_09_09_04_Samples_3000.mat')  # Mein Modell Testdaten (Immer dieselben Testdaten nutzen)

# Daten aufbereiten (r, r_p und phi, phi_p heraussuchen)
r_des_traj = features_training[:, 0]
r_p_des_traj = features_training[:, 2]
phi_des_traj = features_training[:, 1]
phi_p_des_traj = features_training[:, 3]

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
dt = move_time/r_des_traj_divided.shape[1]

# XML Modell laden
xml_name = '2_FHG_Rob_Model_1_data_gen.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reglerparameter setzen
Kp = np.array([[200, 0],
               [0, 200]])

Kd = np.array([[50, 0],
               [0, 50]])

# Listen zum Tracken der Trajektorien
q_vec = []

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

        # kurz warten am Anfang
        time.sleep(1)

        # Simulation beginnen
        for t in range(len(r_des_traj_divided.shape[1])):
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

            # Fehler berechnen
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

            # Bild aktualisieren
            v.sync()

            # Echtzeit synchronisieren
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Echtzeit überschritten bei t={t_vec[t]:.3f}s um {-sleep_time:.4f}s")

        # kurz warten am Ende
        time.sleep(1)

# Messwerte in np array umwandeln
q_vec = np.array(q_vec)
