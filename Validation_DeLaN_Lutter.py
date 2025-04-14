'''
Autor:      Ole Uphaus
Datum:      11.04.2025
Beschreibung:
Dieses skript soll die Validierung des von Lutter gebauten und mit meinen Trainingsdaten trainierten Neuronalen Netzes vornehmen. Dazu wird die ODE einmal mit dem originalen Zustandsraummodell und einmal mit dem DeLaN Netzwerk numerisch gelöst.
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Training_Models.DeLaN_Lutter.DeLaN_model_Lutter import DeepLagrangianNetwork
from pathlib import Path
import os
import torch
from sklearn.preprocessing import StandardScaler

def ODE_2_FHG_Robot(t, x, F_vec, tau_vec, l, m, mL, J):
    dxdt = np.zeros(4)  # 4 Zustände

    # aktuelle Stellgrößen auslesen (Der Zeitvektor ist jeweils in F_vec und tau_vec enthalten -> f_vec(1, :) - Zeitvektor)
    F = np.interp(t, F_vec[0, :], F_vec[1, :])
    tau = np.interp(t, tau_vec[0, :], tau_vec[1, :])

    # Zustandsraumdarstellung
    r = x[0]
    phi = x[1]
    r_p = x[2]
    phi_p = x[3]

    # Ableitungen berechnen
    dxdt[0] = r_p
    dxdt[1] = phi_p
    dxdt[2] = (F - l*m*phi_p**2 + m*phi_p**2*r + mL*phi_p**2*r)/(m + mL)
    dxdt[3] = (tau + r_p*(m*phi_p*(2*l - 2*r) - 2*mL*phi_p*r))/(J + mL*r**2 + m*(l - r)**2)

    return dxdt

def ODE_Neural_Network(t, x, model, F_vec, tau_vec):
    dxdt = np.zeros(4)  # 4 Zustände

    # aktuelle Stellgrößen auslesen (Der Zeitvektor ist jeweils in F_vec und tau_vec enthalten -> f_vec(1, :) - Zeitvektor)
    F = np.interp(t, F_vec[0, :], F_vec[1, :])
    tau = np.interp(t, tau_vec[0, :], tau_vec[1, :])

    # Features erstellen
    features = np.array([x[0], x[1], x[2], x[3], F, tau])
    q = torch.tensor(features[[0, 1]], dtype=torch.float32).unsqueeze(0)
    qd = torch.tensor(features[[2, 3]], dtype=torch.float32).unsqueeze(0)
    tau_delan = torch.tensor(features[[4, 5]], dtype=torch.float32).unsqueeze(0)

    # Prädiktion
    with torch.no_grad():
        outputs = model.inv_dyn(q, qd, tau_delan)
    outputs_numpy = outputs.detach().numpy()

    # Ableitungen
    dxdt[0] = x[2]
    dxdt[1] = x[3]
    dxdt[2] = outputs_numpy[0][0]
    dxdt[3] = outputs_numpy[0][1]

    return dxdt

# Systemparameter
m_kg = 5   # Masse des Arms
mL_kg = 2  # Masse der Last
J_kgm2 = 0.4  # gesamte Rotationsträgheit
l_m = 0.25 # Schwerpunktsabstand (Arm - Last)

# Anfangswerte und Simulationszeit
t_span = [0, 10];    # Simulationszeit

# Anfangswerte
r_0 = 0.5
phi_0 = 0
r_p_0 = 0
phi_p_0 = 0

x_0 = np.array([r_0, phi_0, r_p_0, phi_p_0])    # Vektor der Anfangswerte

# Zeitsignal 
t_u = np.linspace(t_span[0], t_span[1], 1000)

# Eingangssignale
uF_vec = 0 * t_u
utau_vec = np.heaviside(t_u - 3, 1)

# Stellgrößen als 2D-Arrays
F_vec = np.vstack([t_u, uF_vec])
tau_vec = np.vstack([t_u, utau_vec])

# Modell laden
script_path = Path(__file__).resolve()
model_path = os.path.join(script_path.parents[0], "Training_Models", "DeLaN_Lutter", "Saved_Models", "20250414_155654_DeLaN_model.pth")
DeLaN_parameters = torch.load(model_path, weights_only=False)

model = DeepLagrangianNetwork(DeLaN_parameters['n_dof'], **DeLaN_parameters['hyper_param'])
model.load_state_dict(DeLaN_parameters['state_dict'])
model.eval()

# ODE lösen (analytisches Modell)
solution_analytic = solve_ivp(
    ODE_2_FHG_Robot,
    t_span=t_span,
    y0=x_0,
    method='RK23',
    t_eval=t_u,
    args=(F_vec, tau_vec, l_m, m_kg, mL_kg, J_kgm2),
    max_step=0.1
)

# Ergebnisse
t_analytic = solution_analytic.t
r_analytic = solution_analytic.y[0]
phi_analytic = solution_analytic.y[1]

# ODE lösen (Feed-Forward NN)
solution_NN = solve_ivp(
    ODE_Neural_Network,
    t_span=t_span,
    y0=x_0,
    method='RK23',
    t_eval=t_u,
    args=(model, F_vec, tau_vec),
    max_step=0.1
)

# Ergebnisse
t_NN = solution_NN.t
r_NN = solution_NN.y[0]
phi_NN = solution_NN.y[1]

# Plotten der Ergebnisse
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_analytic,r_analytic, label='r_analytic')
plt.plot(t_NN,r_NN, label='r_NN')
plt.title('r(t)')
plt.xlabel('Zeit [s]')
plt.ylabel('r [m]')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_analytic,phi_analytic, label='phi_analytic')
plt.plot(t_NN,phi_NN, label='phi_NN')
plt.title('phi(t)')
plt.xlabel('Zeit [s]')
plt.ylabel('phi [rad]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()