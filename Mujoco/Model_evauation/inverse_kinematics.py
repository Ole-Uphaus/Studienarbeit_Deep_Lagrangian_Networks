'''
Autor:      Ole Uphaus
Datum:     25.05.2025
Beschreibung:
In diesem Skript möchte ich eine Solltrajektorie für mein mujoco modell erstellen. Der Roboter soll die Trajektorie zunächst nur abfahren (ohne Regelung).
'''

import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt

# Parameter Trajektorie (Lemniskate)
amp_x = 0.2
amp_y = 0.1
T = 5
dt = 0.01
omega = 2 * np.pi / T   # Frequenz so anpassen, dass eine Umdrehung in gegebener Zeit durchgeführt wird

offset_x = 1
offset_y = 0.2

# Zeitvektor
t = np.arange(0, T, dt)

# Parametrisierung der 8 (Lemniskate)
x = amp_x * np.sin(omega * t) + offset_x
y = amp_y * np.sin(omega * t) * np.cos(omega * t) + offset_y

# Trajektorie plotten
plt.plot(x, y)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Solltrajektorie - Lemniskate")
plt.xlim(-0.2, 1.5)
plt.ylim(-0.2, 1.5)
plt.grid(True)
plt.show()
