'''
Autor:     Ole Uphaus
Datum:     16.08.2025
Beschreibung:
Mit diesem Skript sollen Plots der Regelergebnisse erstellt werden. Dazu werden die Ergebnisse aus dem entsprechenden Numpy Files geladen.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams

# Verzeichnis mit Hauptversion von DeLaN einbinden (liegt an anderer Stelle im Projekt)
script_path = os.path.dirname(os.path.abspath(__file__))
DeLaN_dir_path = os.path.join(script_path, '..', '..', 'Training_Models', 'DeLaN_Ole')

if DeLaN_dir_path not in sys.path:
    sys.path.insert(0, DeLaN_dir_path)

from DeLaN_functions_Ole import *

# Daten laden
data_path = os.path.join(script_path, 'Control_results_analytic.npy')
results_analytic = np.load(data_path, allow_pickle=True).item()

data_path = os.path.join(script_path, 'Control_results_DeLaN.npy')
results_delan = np.load(data_path, allow_pickle=True).item()

data_path = os.path.join(script_path, 'Control_results_FFNN.npy')
results_ffnn = np.load(data_path, allow_pickle=True).item()

data_path = os.path.join(script_path, 'Control_results_PD.npy')
results_pd = np.load(data_path, allow_pickle=True).item()

# Logische Variablen
save_pdf = True
print_legend = True

# Plots für Studienarbeit
plot_path = r'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\06_Ergebnisse'
plot_1_name = os.path.join(plot_path, 'Ergebnisse_Regelung_xy_Ebene.pdf')
plot_2_name = os.path.join(plot_path, 'Ergebnisse_Regelung_Koordinatenebene.pdf')

# Plot erstellen (ohne Function, da nur einmal verwendet)

# Größe wie in LaTeX (13.75 x 8.5 cm)
cm_to_inch = 1 / 2.54
fig_width = 13.75 * cm_to_inch
fig_height = 8 * cm_to_inch

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
h_soll, = ax.plot(results_analytic['x_des_traj'], results_analytic['y_des_traj'])
h_pd, = ax.plot(results_pd['end_mass_pos_vec'][:, 0], results_pd['end_mass_pos_vec'][:, 1])
ax.plot(results_analytic['end_mass_pos_vec'][:, 0], results_analytic['end_mass_pos_vec'][:, 1])
h_ffnn, = ax.plot(results_ffnn['end_mass_pos_vec'][:, 0], results_ffnn['end_mass_pos_vec'][:, 1])
h_delan, = ax.plot(results_delan['end_mass_pos_vec'][:, 0], results_delan['end_mass_pos_vec'][:, 1])

# Startpunkte plotten
ax.plot(results_pd['end_mass_pos_vec'][0, 0], results_pd['end_mass_pos_vec'][0, 1], marker='o', markersize=4.0,  # klein & dezent
            markerfacecolor='k',
            markeredgecolor='none', zorder=5)

ax.plot(results_analytic['x_des_traj'][0], results_analytic['y_des_traj'][0], marker='o', markersize=4.0,  # klein & dezent
            markerfacecolor='k',
            markeredgecolor='none', zorder=5)

# Pfeile plotten
ax.annotate(
    "", xy=(results_analytic['x_des_traj'][70], results_analytic['y_des_traj'][70]), xytext=(results_analytic['x_des_traj'][69], results_analytic['y_des_traj'][69]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_soll.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_pd['end_mass_pos_vec'][150, 0], results_pd['end_mass_pos_vec'][150, 1]), xytext=(results_pd['end_mass_pos_vec'][149, 0], results_pd['end_mass_pos_vec'][149, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_pd.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_pd['end_mass_pos_vec'][700, 0], results_pd['end_mass_pos_vec'][700, 1]), xytext=(results_pd['end_mass_pos_vec'][699, 0], results_pd['end_mass_pos_vec'][699, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_pd.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_pd['end_mass_pos_vec'][1750, 0], results_pd['end_mass_pos_vec'][1750, 1]), xytext=(results_pd['end_mass_pos_vec'][1749, 0], results_pd['end_mass_pos_vec'][1749, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_pd.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_ffnn['end_mass_pos_vec'][100, 0], results_ffnn['end_mass_pos_vec'][100, 1]), xytext=(results_ffnn['end_mass_pos_vec'][99, 0], results_ffnn['end_mass_pos_vec'][99, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_ffnn.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_delan['end_mass_pos_vec'][500, 0], results_delan['end_mass_pos_vec'][500, 1]), xytext=(results_delan['end_mass_pos_vec'][499, 0], results_delan['end_mass_pos_vec'][499, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_delan.get_color()),
    zorder=6
)

ax.annotate(
    "", xy=(results_delan['end_mass_pos_vec'][1850, 0], results_delan['end_mass_pos_vec'][1850, 1]), xytext=(results_delan['end_mass_pos_vec'][1849, 0], results_delan['end_mass_pos_vec'][1849, 1]),
    arrowprops=dict(arrowstyle="-|>",
                    shrinkA=0, shrinkB=0,
                    linewidth=1,
                    mutation_scale=11,
                    color=h_delan.get_color()),
    zorder=6
)

# Achseneinstellungen
# ax.set_xlim((seed_loss_vec[0, 0] - 1), seed_loss_vec[-1, 0])
ax.set_ylabel('y')
ax.set_title('')
ax.grid(True)
ax.set_facecolor('white')
ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

# Achsgrenzen mit Puffer (Manuell verändert)
yl = ax.get_ylim()
dy = yl[1] - yl[0]
new_ylim = [yl[0] - 0.05 * dy, yl[1] + 0.05 * dy]
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
        ['Solltraj.', 'PD', 'Analyt.', 'MLP', 'DeLaN'],
        loc='upper left',
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        fancybox=False,
        borderpad=0.3,
        handletextpad=0.5,
        labelspacing=0.2
    )
    legend.get_frame().set_linewidth(line_thickness)

ax.set_xlabel('x')

fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.15)

if save_pdf:
    fig.savefig(plot_1_name, transparent=True, format='pdf')

plt.show()

# Zweiter Plot
zeros_vec = np.zeros_like(results_analytic['t_vec'])

quad_subplot(
        results_analytic['t_vec'],
        [np.concatenate([results_analytic['r_des_traj'].reshape(-1, 1), results_pd['q_vec'][:, 1].reshape(-1, 1), results_analytic['q_vec'][:, 1].reshape(-1, 1), results_ffnn['q_vec'][:, 1].reshape(-1, 1), results_delan['q_vec'][:, 1].reshape(-1, 1)], axis=1).reshape(-1, 5), 
        np.concatenate([zeros_vec.reshape(-1, 1), results_pd['error_r'].reshape(-1, 1), results_analytic['error_r'].reshape(-1, 1), results_ffnn['error_r'].reshape(-1, 1), results_delan['error_r'].reshape(-1, 1)], axis=1).reshape(-1, 5),
        np.concatenate([results_analytic['phi_des_traj'].reshape(-1, 1), results_pd['q_vec'][:, 0].reshape(-1, 1), results_analytic['q_vec'][:, 0].reshape(-1, 1), results_ffnn['q_vec'][:, 0].reshape(-1, 1), results_delan['q_vec'][:, 0].reshape(-1, 1)], axis=1).reshape(-1, 5), 
        np.concatenate([zeros_vec.reshape(-1, 1), results_pd['error_phi'].reshape(-1, 1), results_analytic['error_phi'].reshape(-1, 1), results_ffnn['error_phi'].reshape(-1, 1), results_delan['error_phi'].reshape(-1, 1)], axis=1).reshape(-1, 5)],
        r'$t \, / \, \mathrm{s}$',
        [r'$r_{RS} \, / \, \mathrm{m}$',r'$e_{r} \, / \, \mathrm{m}$',r'$\varphi_{RS} \, / \, \mathrm{m}$',r'$e_{\varphi} \, / \, \mathrm{rad}$'],
        ['', '', '', ''],
        [[], [], [], ['Solltraj.', 'PD', 'Analyt.', 'MLP', 'DeLaN']],
        plot_2_name,
        True,
        True
    )