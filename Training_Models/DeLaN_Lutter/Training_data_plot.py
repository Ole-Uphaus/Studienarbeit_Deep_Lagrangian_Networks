'''
Autor:      Ole Uphaus
Datum:      15.04.2025
Beschreibung:
In diesem Skript will ich die Trainingsdaten von plotten, die Lutter für das Training seines Modells verwendet hat. Vielleicht geben diese Daten etwas mehr Aufschluss.
'''

import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import os
import scipy.io

def load_dataset(n_characters=3, filename="D:\\Programmierung_Ole\\Studienarbeit_Deep_Lagrangian_Networks\\Training_Models\\DeLaN_Lutter\\character_data.pickle", test_label=("e", "q", "v")):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    n_dof = 2

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    # test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean

# Datensatz laden
train_data, test_data, divider, dt_mean = load_dataset()    # Buchstaben Modell (Lutter)
train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau = train_data
test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g = test_data

# Anzahl der Buchstaben aussuchen
print('Train Characters: ', train_labels)
print('Num. Characters: ', len(train_labels))

# Plotten Trainingsdaten Lutter
samples_vec = np.arange(1, train_qp.shape[0] + 1)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(samples_vec, train_qp[:, 0], label='q1')
plt.title('q1')
plt.xlabel('Samples')
plt.ylabel('q [rad]')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(samples_vec, train_qp[:, 1], label='q2')
plt.title('q2')
plt.xlabel('Zeit [s]')
plt.ylabel('q [rad]')
plt.grid(True)
plt.legend()

plt.show()