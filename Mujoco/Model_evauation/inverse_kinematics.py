'''
Autor:      Ole Uphaus
Datum:     25.05.2025
Beschreibung:
In diesem Skript möchte ich eine Solltrajektorie für mein mujoco modell erstellen. Der Roboter soll die Trajektorie zunächst nur abfahren (ohne Regelung).
'''

import mujoco
from mujoco import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# In diesem Fall lässt sich die inverse Kinematik eindeutig auflösen
def inverse_kinematics_2_DOF(x_e, y_e):
    # Berechnung von phi
    phi = np.arctan2(y_e, x_e)

    # Berechnung von r
    r = x_e/np.cos(phi)

    return r, phi

# Parameter Trajektorie (Lemniskate)
amp_x = 0.2
amp_y = 0.1
T = 5
dt = 0.01
omega = 2 * np.pi / T   # Frequenz so anpassen, dass eine Umdrehung in gegebener Zeit durchgeführt wird

offset_x = 1
offset_y = 0.2

# Zeitvektor
t_vec = np.arange(0, T, dt)

# Parametrisierung der 8 (Lemniskate)
x_traj = amp_x * np.sin(omega * t_vec) + offset_x
y_traj = amp_y * np.sin(omega * t_vec) * np.cos(omega * t_vec) + offset_y

# # Trajektorie plotten
# plt.plot(x_traj, y_traj)
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.title("Solltrajektorie - Lemniskate")
# plt.xlim(-0.2, 1.5)
# plt.ylim(-0.2, 1.5)
# plt.grid(True)
# plt.show()

# XML Modell laden
xml_name = '2_FHG_Rob_Model_1.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Anzahl an wiederholungen
rep = 1

# Initiale Position einnehmen
r_init, phi_init = inverse_kinematics_2_DOF(x_traj[0], y_traj[0])

data.qpos[0] = phi_init
data.qpos[1] = r_init

# Viewer starten
with viewer.launch_passive(model, data) as v:
    # Kamera konfigurieren
    v.cam.lookat[:] = [0.5, 0, 0]   # Zentrum deiner Szene
    v.cam.distance = 3.0                # Nähe
    v.cam.azimuth = 90
    v.cam.elevation = -70

    # kurz warten am Anfang
    time.sleep(1)

    for _ in range(rep):
        for t in range(len(t_vec)):
            # inverse Kinematik auswerten
            r, phi = inverse_kinematics_2_DOF(x_traj[t], y_traj[t])

            # Gelenkpositionen setzen
            data.qpos[0] = phi
            data.qpos[1] = r

            # Kinematik berechnen
            mujoco.mj_forward(model, data)

            # Bild aktualisieren
            v.sync()

            # Zeitschritt abwarten
            time.sleep(dt)

    # kurz warten am Ende
    time.sleep(1)
