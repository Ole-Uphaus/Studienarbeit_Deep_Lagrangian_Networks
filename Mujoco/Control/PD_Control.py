'''
Autor:      Ole Uphaus
Datum:     05.06.2025
Beschreibung:
In diesem Skript Werde ich für das Mujoco modell eine einfache PD-Folgeregelung entwerfen.
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

# XML Modell laden
xml_name = '2_FHG_Rob_Model_1.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initiale Position einnehmen
r_init, phi_init = inverse_kinematics_2_DOF(x_traj[0], y_traj[0])

data.qpos[0] = phi_init
data.qpos[1] = r_init

# Reglerparameter setzen
Kp_phi = 10
Kp_r = 10

Kd_phi = 1
Kd_r = 1

# Listen zum Tracken der Trajektorien
end_mass_pos_vec = []
q_des_vec = []
q_vec = []

# Viewer starten
with viewer.launch_passive(model, data) as v:
    # Kamera konfigurieren
    v.cam.lookat[:] = [0.5, 0, 0]   # Zentrum deiner Szene
    v.cam.distance = 3.0                # Nähe
    v.cam.azimuth = 90
    v.cam.elevation = -70

    # kurz warten am Anfang
    time.sleep(1)

    # Simulation beginnen
    for t in range(len(t_vec)):
        # inverse Kinematik auswerten
        r_des, phi_des = inverse_kinematics_2_DOF(x_traj[t], y_traj[t])

        # Ist-Werte auslesen
        phi = data.qpos[0]
        r = data.qpos[1]

        phi_dot = data.qvel[0]
        r_dot = data.qvel[1]

        # PD-Regelung berechnen
        tau_phi = Kp_phi*(phi_des - phi) - Kd_phi*phi_dot
        tau_r = Kp_r*(r_des - r) - Kd_r*r_dot

        # Steuergrößen setzen
        data.ctrl[0] = tau_phi 
        data.ctrl[1] = tau_r    

        # Kinematik und Dynamik berechnen
        mujoco.mj_step(model, data)

        # Messwerte abspeichern
        # Endeffektor Position auslesen
        end_mass_pos_vec.append(data.site_xpos[0].copy())

        # Sollgelenkwinkel auslesen
        q_des_vec.append([phi_des, r_des])

        # Istgelenkwonkel auslesen
        q_vec.append(data.qpos.copy())

        # Bild aktualisieren
        v.sync()

        # Zeitschritt abwarten
        time.sleep(dt)

    # kurz warten am Ende
    time.sleep(1)

# Messwerte in np array umwandeln
end_mass_pos_vec = np.array(end_mass_pos_vec)
q_des_vec = np.array(q_des_vec)
q_vec = np.array(q_vec)

# Soll vs. Ist Trajektorie plotten
plt.figure()

plt.plot(x_traj, y_traj, label="Solltrajektorie")
plt.plot(end_mass_pos_vec[:, 0], end_mass_pos_vec[:, 1], label="Ist-Trajektorie")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Solltrajektorie - Lemniskate")
plt.xlim(-0.2, 1.5)
plt.ylim(-0.2, 0.6)
plt.grid(True)
plt.legend()

# Soll vs. Ist Gelenkkoordinaten
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(t_vec, q_des_vec[:, 0], label='phi soll')
plt.plot(t_vec, q_vec[:, 0] ,label='phi ist')
plt.title('phi')
plt.xlabel('t')
plt.ylabel('phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_vec, q_des_vec[:, 1], label='r soll')
plt.plot(t_vec, q_vec[:, 1] ,label='r ist')
plt.title('r')
plt.xlabel('t')
plt.ylabel('r')
plt.grid(True)
plt.legend()

plt.show()