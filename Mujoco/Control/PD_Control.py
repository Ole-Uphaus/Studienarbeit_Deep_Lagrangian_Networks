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

# Parameter Trajektorie 
amp_phi = 0.1
amp_r = 0.2
T = 5
dt = 0.002
omega = 2 * np.pi / T   # Frequenz so anpassen, dass eine Umdrehung in gegebener Zeit durchgeführt wird

offset_phi = 0.2
offset_r = 1.0

# Zeitvektor
t_vec = np.arange(0, T, dt)

# Trajektorien für r und phi festlegen (+ Ableitung)
phi_des_traj = amp_phi * np.sin(2 * omega * t_vec) + offset_phi
r_des_traj = amp_r * np.sin(omega * t_vec) + offset_r

phi_p_des_traj = amp_phi * 2 * omega * np.cos(2 * omega * t_vec)
r_p_des_traj = amp_r * omega * np.cos(omega * t_vec)

# x und y Koordinaten ableiten
x_des_traj = r_des_traj * np.cos(phi_des_traj)
y_des_traj = r_des_traj * np.sin(phi_des_traj)

# XML Modell laden
xml_name = '2_FHG_Rob_Model_1.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Sites auf die Solltrajektorie setzen
anz_sites = 500
site_counter = 0
site_step = phi_p_des_traj.shape[0] // anz_sites

for i in range(len(t_vec)):
    if np.mod(i, site_step) == 0:
        # Site Position aktualisieren
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'sitebody_des_{site_counter}')
        model.body_pos[body_id] = np.array([x_des_traj[i], y_des_traj[i], 0])

        # Counter aktualisieren
        site_counter += 1

mujoco.mj_forward(model, data)

# Initiale Position einnehmen
data.qpos[0] = phi_des_traj[0] + np.pi/40
data.qpos[1] = r_des_traj[0] - 0.05

# Reglerparameter setzen
Kp = np.array([[200, 0],
               [0, 200]])

Kd = np.array([[50, 0],
               [0, 50]])

# Listen zum Tracken der Trajektorien
end_mass_pos_vec = []
q_vec = []

# ID des end effector sites abspeichern
end_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f'end_mass_site')
site_counter = 0

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
        # Startzeit
        loop_start = time.time()

        # Sollwerte auslesen
        phi_des = phi_des_traj[t]
        r_des = r_des_traj[t]

        phi_p_des = phi_p_des_traj[t]
        r_p_des = r_p_des_traj[t]

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
        # Endeffektor Position auslesen
        end_mass_pos_vec.append(data.site_xpos[end_site_id].copy())

        # Istgelenkwinkel auslesen
        q_vec.append(data.qpos.copy())

        # Trajektorie malen
        if np.mod(t, site_step) == 0:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'sitebody_ist_{site_counter}')
            model.body_pos[body_id] = data.site_xpos[end_site_id].copy()

            # Counter aktualisieren
            site_counter += 1
            mujoco.mj_forward(model, data)

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
end_mass_pos_vec = np.array(end_mass_pos_vec)
q_vec = np.array(q_vec)

# Trajektorienfolgefehler berechnen
error_phi = phi_des_traj - q_vec[:, 0]
error_r = r_des_traj - q_vec[:, 1]

# Mittelwert der quadrierten Fehler (MSE)
mse_phi = np.mean(error_phi**2)
mse_r = np.mean(error_r**2)

# Ausgabe
print(f"MSE für phi: {mse_phi:.4e}")
print(f"MSE für r: {mse_r:.4e}")

# Soll vs. Ist Trajektorie plotten
plt.figure()

plt.plot(x_des_traj, y_des_traj, label="Solltrajektorie")
plt.plot(end_mass_pos_vec[:, 0], end_mass_pos_vec[:, 1], label="Ist-Trajektorie")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Solltrajektorie")
plt.xlim(0.4, 1.4)
plt.ylim(-0.1, 0.5)
plt.grid(True)
plt.legend()

# Soll vs. Ist Gelenkkoordinaten
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(t_vec, phi_des_traj, label='phi soll')
plt.plot(t_vec, q_vec[:, 0], label='phi ist')
plt.title('phi')
plt.xlabel('t')
plt.ylabel('phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_vec, r_des_traj, label='r soll')
plt.plot(t_vec, q_vec[:, 1], label='r ist')
plt.title('r')
plt.xlabel('t')
plt.ylabel('r')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_vec, error_phi, label='error phi')
plt.title('Folgefehler phi')
plt.xlabel('t')
plt.ylabel('phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t_vec, error_r, label='error r')
plt.title('Folgefehler r')
plt.xlabel('t')
plt.ylabel('r')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()