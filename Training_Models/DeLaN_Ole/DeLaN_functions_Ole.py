'''
Autor:      Ole Uphaus
Datum:     05.05.2025
Beschreibung:
Dieses Skript enthält Funktionen im Zusammenhang mit dem Training und der Erprobung von Deep Lagrangien Networks.
'''

import scipy.io
import os
import numpy as np
import torch
from matplotlib import rcParams
import matplotlib.pyplot as plt

def extract_training_data(file_name, target_folder):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', target_folder, file_name)

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

def model_evaluation(model, q_test, qd_test, qdd_test, tau_test, use_inverse_model, use_forward_model, use_energy_consumption):
    # Forward pass
    out_eval = model(q_test, qd_test, qdd_test)    # Inverses Modell
    qdd_hat, _, _, _ = model.forward_dynamics(q_test, qd_test, tau_test) # Vorwärts Modell

    tau_hat_eval = out_eval[0].cpu().detach().numpy()   # Tesnoren auf cpu legen, gradienten entfernen, un numpy arrays umwandeln
    H_eval = out_eval[1].cpu().detach().numpy()
    c_eval = out_eval[2].cpu().detach().numpy()
    g_eval = out_eval[3].cpu().detach().numpy()
    tau_fric_eval = out_eval[4].cpu().detach().numpy()
    T_dt = out_eval[6]
    V_dt = out_eval[7]

    # Loss initialisieren
    test_loss = np.array(0.0)

    if use_inverse_model:
        # Fehler aus inverser Dynamik berechnen (Schätzung von tau)
        err_inv_dyn_test = np.sum((tau_hat_eval - tau_test.cpu().detach().numpy())**2, axis=1)
        mean_err_inv_dyn_eval = np.mean(err_inv_dyn_test)

        # Test Loss berechnen
        test_loss += mean_err_inv_dyn_eval

    if use_forward_model:
        # Fehler aus Vorwärtsmodell berechnen (Schätzung von qdd)
        err_for_dyn_test = np.sum((qdd_hat.cpu().detach().numpy() - qdd_test.cpu().detach().numpy())**2, axis=1)
        mean_err_for_dyn_eval = np.mean(err_for_dyn_test)

        # Test Loss berechnen
        test_loss += mean_err_for_dyn_eval

    if use_energy_consumption:
        # Fehler aus Energieerhaltung berechnen
        E_dt_mot = torch.einsum('bi,bi->b', qd_test, tau_test)
        E_dt_mot_hat = T_dt + V_dt + torch.einsum('bi,bi->b', qd_test, out_eval[4])  

        err_E_dt_mot = (E_dt_mot_hat.cpu().detach().numpy() - E_dt_mot.cpu().detach().numpy())**2
        mean_err_E_dt_mot = np.mean(err_E_dt_mot)

        # Test Loss berechnen
        test_loss += mean_err_E_dt_mot

    if use_inverse_model == False and use_forward_model == False and use_energy_consumption == False:
        raise ValueError("Ungültige Konfiguration: 'use_inverse_model' und 'use_forward_model' dürfen nicht beide False sein.")

    return test_loss, tau_hat_eval, H_eval, c_eval, g_eval, tau_fric_eval

def eval_friction_graph(model, device):
    # Geschwindigkeitstensoren definieren
    qd = torch.concatenate((torch.linspace(0, 10, 1000).view((-1, 1)), torch.linspace(0, 10, 1000).view((-1, 1))), dim=1).to(device)

    # Modell auswerten
    _, _, _, _, tau_fric, _, _, _ = model(torch.zeros_like(qd), qd, torch.zeros_like(qd))

    # Tensoren in numpy umwandeln für Plot
    qd_numpy = qd[:, 1].cpu().detach().numpy()
    tau_fric_numpy = tau_fric.cpu().detach().numpy()

    return qd_numpy, tau_fric_numpy

# Doppelter subplot
def double_subplot(x, y_list, xlabel_str, ylabel_str_list, title_str_list, legend_label_list, filename, save_pdf=True, print_legend=True):
    
    # Größe wie in LaTeX (13.75 x 8.5 cm)
    cm_to_inch = 1 / 2.54
    fig_width = 13.75 * cm_to_inch
    fig_height = 8.5 * cm_to_inch

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

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=300)

    for i, ax in enumerate(axes):
        y_data = y_list[i]  # y_data: (n_samples, n_signals)
        for k in range(y_data.shape[1]):
            ax.plot(x, y_data[:, k], linewidth=1.5)

        # x-Achse vollständig nutzen
        ax.set_xlim(x[0], x[-1])

        ax.set_ylabel(ylabel_str_list[i])
        ax.set_title(title_str_list[i])
        ax.grid(True)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

        # Achsgrenzen mit Puffer
        yl = ax.get_ylim()
        dy = yl[1] - yl[0]
        new_ylim = [yl[0] - 0.05 * dy, yl[1] + 0.05 * dy]
        ax.set_ylim(new_ylim[0], new_ylim[1])

        # Nullinie etwas dicker machen
        if (new_ylim[0] < 0) and (new_ylim[1] > 0):
            ax.axhline(y=0, color=(0.8, 0.8, 0.8), linewidth=0.7, zorder=0)

        # liniendicke
        ax.grid(True, linewidth=line_thickness, color=(0.8, 0.8, 0.8))  # z. B. 0.3 für feine Gridlines
        ax.spines['top'].set_linewidth(line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        ax.spines['right'].set_linewidth(line_thickness)

        if print_legend:
            legend = ax.legend(legend_label_list[i], # Legende wie in matlab
                loc='upper right',
                frameon=True,         # Rahmen anzeigen (MATLAB-like)
                edgecolor='black',    # Rahmenfarbe
                framealpha=1.0,       # Kein transparenter Hintergrund
                fancybox=False,       # Kein abgerundeter Rahmen
                borderpad=0.3,        # Weniger Padding im Rahmen
                handletextpad=0.5     # Abstand zwischen Linie und Text
                )
            legend.get_frame().set_linewidth(line_thickness)

    # x-Achse setzen
    axes[1].set_xlabel(xlabel_str)

    fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.13, hspace=0.35)

    if save_pdf:
        fig.savefig(filename, transparent=True, format='pdf')

    plt.show()


# Vierfacher subplot
def quad_subplot(x, y_list, xlabel_str, ylabel_str_list, title_str_list, legend_label_list, filename, save_pdf=True, print_legend=True):
    
    # Größe wie in LaTeX (13.75 x 8.5 cm)
    cm_to_inch = 1 / 2.54
    fig_width = 13.75 * cm_to_inch
    fig_height = 8.5 * cm_to_inch

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

    # 2 Zeilen × 2 Spalten
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=300)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        y_data = y_list[i]  # y_data: (n_samples, n_signals)
        for k in range(y_data.shape[1]):
            ax.plot(x, y_data[:, k], linewidth=1.5)

        # x-Achse vollständig nutzen
        ax.set_xlim(x[0], x[-1])

        ax.set_ylabel(ylabel_str_list[i])
        ax.set_title(title_str_list[i])
        ax.grid(True)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

        # Achsgrenzen mit Puffer
        yl = ax.get_ylim()
        dy = yl[1] - yl[0]
        new_ylim = [yl[0] - 0.05 * dy, yl[1] + 0.05 * dy]
        ax.set_ylim(new_ylim[0], new_ylim[1])

        # Nullinie etwas dicker machen
        if (new_ylim[0] < 0) and (new_ylim[1] > 0):
            ax.axhline(y=0, color=(0.8, 0.8, 0.8), linewidth=0.7, zorder=0)

        # liniendicke
        ax.grid(True, linewidth=line_thickness, color=(0.8, 0.8, 0.8))  # z. B. 0.3 für feine Gridlines
        ax.spines['top'].set_linewidth(line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        ax.spines['right'].set_linewidth(line_thickness)

        if print_legend:
            legend = ax.legend(legend_label_list[i], # Legende wie in matlab
                loc='upper right',
                frameon=True,         # Rahmen anzeigen (MATLAB-like)
                edgecolor='black',    # Rahmenfarbe
                framealpha=1.0,       # Kein transparenter Hintergrund
                fancybox=False,       # Kein abgerundeter Rahmen
                borderpad=0.3,        # Weniger Padding im Rahmen
                handletextpad=0.5     # Abstand zwischen Linie und Text
                )
            legend.get_frame().set_linewidth(line_thickness)

    # x-Achse setzen
    for ax in axes[-2:]:
        ax.set_xlabel(xlabel_str)

    fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.12, hspace=0.45, wspace=0.35)
    # plt.tight_layout()

    if save_pdf:
        fig.savefig(filename, transparent=True, format='pdf')

    plt.show()

# Doppelter subplot mit variablen x-Achsenbeschriftungen 
def double_subplot_varx(x_list, y_list, xlabel_str_list, ylabel_str_list, title_str_list, legend_label_list, filename, save_pdf=True, print_legend=True):
    
    # Größe wie in LaTeX (13.75 x 8.5 cm)
    cm_to_inch = 1 / 2.54
    fig_width = 13.75 * cm_to_inch
    fig_height = 8.5 * cm_to_inch

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

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=300)

    for i, ax in enumerate(axes):
        y_data = y_list[i]  # y_data: (n_samples, n_signals)
        for k in range(y_data.shape[1]):
            ax.plot(x_list[i], y_data[:, k], linewidth=1.5)

        # x-Achse vollständig nutzen
        ax.set_xlim(x_list[i][0], x_list[i][-1])

        ax.set_xlabel(xlabel_str_list[i])
        ax.set_ylabel(ylabel_str_list[i])
        ax.set_title(title_str_list[i])
        ax.grid(True)
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

        # Achsgrenzen mit Puffer
        yl = ax.get_ylim()
        dy = yl[1] - yl[0]
        new_ylim = [yl[0] - 0.05 * dy, yl[1] + 0.05 * dy]
        ax.set_ylim(new_ylim[0], new_ylim[1])

        # Nullinie etwas dicker machen
        if (new_ylim[0] < 0) and (new_ylim[1] > 0):
            ax.axhline(y=0, color=(0.8, 0.8, 0.8), linewidth=0.7, zorder=0)

        # liniendicke
        ax.grid(True, linewidth=line_thickness, color=(0.8, 0.8, 0.8))  # z. B. 0.3 für feine Gridlines
        ax.spines['top'].set_linewidth(line_thickness)
        ax.spines['bottom'].set_linewidth(line_thickness)
        ax.spines['left'].set_linewidth(line_thickness)
        ax.spines['right'].set_linewidth(line_thickness)

        if print_legend:
            legend = ax.legend(legend_label_list[i], # Legende wie in matlab
                loc='upper right',
                frameon=True,         # Rahmen anzeigen (MATLAB-like)
                edgecolor='black',    # Rahmenfarbe
                framealpha=1.0,       # Kein transparenter Hintergrund
                fancybox=False,       # Kein abgerundeter Rahmen
                borderpad=0.3,        # Weniger Padding im Rahmen
                handletextpad=0.5     # Abstand zwischen Linie und Text
                )
            legend.get_frame().set_linewidth(line_thickness)

    fig.subplots_adjust(left=0.13, right=0.91, top=0.92, bottom=0.13, hspace=0.45)

    if save_pdf:
        fig.savefig(filename, transparent=True, format='pdf')

    plt.show()

# Einzelner Plot logarithmisch
def single_plot_log(x, y_data, xlabel_str, ylabel_str, title_str, legend_label_list, filename, save_pdf=True, print_legend=True):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Größe wie in LaTeX (13.75 x 8.5 cm)
    cm_to_inch = 1 / 2.54
    fig_width = 13.75 * cm_to_inch
    fig_height = 8.5 * cm_to_inch

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

    # y_data: (n_samples, n_signals)
    for k in range(y_data.shape[1]):
        ax.plot(x, y_data[:, k], linewidth=1.5)

    # Achseneinstellungen
    ax.set_yscale("log")    # Logarithmisch
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(ylabel_str)
    ax.set_title(title_str)
    ax.grid(True)
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=4, width=line_thickness)

    # Achsgrenzen mit Puffer
    yl = ax.get_ylim()
    dy = yl[1] - yl[0]
    new_ylim = [yl[0] - 0.05 * dy, yl[1] + 0.05 * dy]
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
            legend_label_list,
            loc='upper right',
            frameon=True,
            edgecolor='black',
            framealpha=1.0,
            fancybox=False,
            borderpad=0.3,
            handletextpad=0.5
        )
        legend.get_frame().set_linewidth(line_thickness)

    ax.set_xlabel(xlabel_str)

    fig.subplots_adjust(left=0.13, right=0.91, top=0.93, bottom=0.13)

    if save_pdf:
        fig.savefig(filename, transparent=True, format='pdf')

    plt.show()