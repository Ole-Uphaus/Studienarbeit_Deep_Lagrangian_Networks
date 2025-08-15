'''
Autor:     Ole Uphaus
Datum:     14.08.2025
Beschreibung:
Mit diesem Skript soll ein Plot der Seed Untersuchung erstellt werden. Dazu werden die Ergebnisse aus dem entsprechenden Numpy File geladen.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'Ergebnisse_Seed_Untersuchung.npy')

# Ergebnisvektor laden
seed_loss_vec = np.load(data_path)
print(seed_loss_vec)

# Logische Variablen
save_pdf = True
print_legend = True

# Plots für Studienarbeit
plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
plot_1_name = os.path.join(plot_path, 'Seed_Sensitivitaet.pdf')

# Plot erstellen (ohne Function, da nur einmal verwendet)

# Größe wie in LaTeX (13.75 x 8.5 cm)
cm_to_inch = 1 / 2.54
fig_width = 13.75 * cm_to_inch
fig_height = 6 * cm_to_inch

# Liniendicke
line_thickness = 0.4

# LaTeX Einstellungen
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,  
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9
})

fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=300)

# Plotten
ax.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 1])
ax.scatter(seed_loss_vec[:, 0], seed_loss_vec[:, 2])

# Achseneinstellungen
ax.set_yscale("log")    # Logarithmisch
ax.set_xlim((seed_loss_vec[0, 0] - 1), seed_loss_vec[-1, 0])
ax.set_ylabel('Loss')
ax.set_title('')
ax.grid(True)
ax.set_facecolor('white')
ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

# Achsgrenzen mit Puffer (Manuell angepasst)
yl = ax.get_ylim()
dy = yl[1] - yl[0]
new_ylim = [yl[0], yl[1]]
new_ylim[1] = 10
if new_ylim[0] > 0 and new_ylim[1] > 0:
    ax.set_ylim(new_ylim[0], new_ylim[1])

# Nullinie etwas dicker
if (new_ylim[0] < 0) and (new_ylim[1] > 0):
    ax.axhline(y=0, color=(0.8, 0.8, 0.8), linewidth=0.7, zorder=0)

# Linienbreiten für Rahmen und Grid
ax.grid(True, linewidth=line_thickness, color=(0.8, 0.8, 0.8))
for spine in ax.spines.values():
    spine.set_linewidth(line_thickness)

if print_legend:
    legend = ax.legend(
        ['Training', 'Test'],
        loc='upper right',
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        fancybox=False,
        borderpad=0.3,
        handletextpad=0.5
    )
    legend.get_frame().set_linewidth(line_thickness)

ax.set_xlabel('Seed')

fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.21)

if save_pdf:
    fig.savefig(plot_1_name, transparent=True, format='pdf')

plt.show()