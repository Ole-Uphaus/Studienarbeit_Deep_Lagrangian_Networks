'''
Autor:      Ole Uphaus
Datum:     24.05.2025
Beschreibung:
In diesem Skript möchte ich die Mujoco Python Schnittstelle ausprobieren. Dazu werde ich versuchen die inverse Dynamik meines Modells auszuwerten und diese mit dem selbst hergeleiteten analytischen Modell zu vergleichen.
'''

import mujoco
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

# Funktion aus dem DeLaN Training zum laden der Trainingsdaten
def extract_training_data(file_name):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', 'MATLAB_Simulation', file_name)

    # Pfad normieren
    data_path = os.path.normpath(data_path)

    # Daten extrahieren
    data = scipy.io.loadmat(data_path)

    features_training = data['features_training']
    labels_training = data['labels_training']
    features_test = data['features_test']
    labels_test = data['labels_test']
    Mass_Cor_test = data['Mass_Cor_test']

    # Zusammensetzung der vektoren ändern, da Erstellung in Matlab für Inverse Dynamik ausgelegt war
    features_training_delan = np.concatenate((features_training[:, :4], labels_training), axis=1)   # (q, qp, qpp)
    features_test_delan = np.concatenate((features_test[:, :4], labels_test), axis=1)   

    labels_training_delan = features_training[:, 4:]
    labels_test_delan = features_test[:, 4:]

    return features_training_delan, labels_training_delan, features_test_delan, labels_test_delan, Mass_Cor_test

# Mit Matlab erzeugte Trainingsdaten laden
features_training, labels_training, _, _, _ = extract_training_data('SimData_V3_Rob_Model_1_2025_05_09_10_26_33_Samples_1500.mat')

# Daten zuorden (q = [r, phi])
q_matlab = features_training[:, [0, 1]]
q_p_matlab = features_training[:, [2, 3]]
q_pp_matlab = features_training[:, [4, 5]]
tau_matlab = labels_training

# XML Modell laden
xml_name = '2_FHG_Rob_Model_1.xml'
script_path = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_path, '..', 'Models', xml_name)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Inverse Dynamik mit Mujoco Modell auswerten (Aufpassen weil q_matlab = [r, phi] und q_mujoco = [phi, r])
tau_mujoco = []
for i in range(q_matlab.shape[0]):
    # Position
    data.qpos[0] = q_matlab[i, 1] 
    data.qpos[1] = q_matlab[i, 0]

    # Geschwindigkeit
    data.qvel[0] = q_p_matlab[i, 1] 
    data.qvel[1] = q_p_matlab[i, 0]

    # Geschwindigkeit
    data.qacc[0] = q_pp_matlab[i, 1] 
    data.qacc[1] = q_pp_matlab[i, 0]

    # Inverses Modell auswerten
    mujoco.mj_inverse(model, data)

    # Auswertung der inversen Dynamik speichern (hier wieder Vertauschung vornehmen)
    tau_mujoco.append([data.qfrc_inverse.copy()[1], data.qfrc_inverse.copy()[0]])

tau_mujoco = np.array(tau_mujoco)

# Verläufe plotten
samples_vec = np.arange(1, q_matlab.shape[0] + 1)

# q
plt.figure()

plt.subplot(2, 3, 1)
plt.plot(samples_vec, q_matlab[:, 0], label='q1')
plt.title('q1')
plt.xlabel('Samples')
plt.ylabel('q1')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(samples_vec, q_matlab[:, 1], label='q2')
plt.title('q2')
plt.xlabel('Samples')
plt.ylabel('q2')
plt.grid(True)
plt.legend()

# q_p
plt.subplot(2, 3, 2)
plt.plot(samples_vec, q_p_matlab[:, 0], label='qp1')
plt.title('qp1')
plt.xlabel('Samples')
plt.ylabel('qp1')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(samples_vec, q_p_matlab[:, 1], label='qp2')
plt.title('qp2')
plt.xlabel('Samples')
plt.ylabel('qp2')
plt.grid(True)
plt.legend()


# q_pp
plt.subplot(2, 3, 3)
plt.plot(samples_vec, q_pp_matlab[:, 0], label='qpp1')
plt.title('qpp1')
plt.xlabel('Samples')
plt.ylabel('qpp1')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(samples_vec, q_pp_matlab[:, 1], label='qpp2')
plt.title('qpp2')
plt.xlabel('Samples')
plt.ylabel('qpp2')
plt.grid(True)
plt.legend()

plt.tight_layout()

# tau
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(samples_vec, tau_matlab[:, 0], label='tau1 matlab')
plt.plot(samples_vec, tau_mujoco[:, 0], label='tau1 mujoco')
plt.title('tau1')
plt.xlabel('Samples')
plt.ylabel('tau1')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(samples_vec, tau_matlab[:, 1], label='tau2 matlab')
plt.plot(samples_vec, tau_mujoco[:, 1], label='tau2 mujoco')
plt.title('tau2')
plt.xlabel('Samples')
plt.ylabel('tau2')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()