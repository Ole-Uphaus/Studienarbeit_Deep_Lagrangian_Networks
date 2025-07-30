% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      01.07.2025
% Beschreibung:
% In diesem Skript werde ich auf Basis des Torsionsschwingermodells
% Trainingsdaten für das DeLaN Modell generieren. Dazu werde ich zunächst
% trajektorien für eine Scheibe (1 FHG System) generieren und die
% benötigten Drehmomente berechnen. Anschließend werde ich mithilfe der
% berechneten Drehmomente eine Simulation der 2 FHG-Systems durchführen.
% -------------------------------------------------------------

clc
clear
close all

% Diese Funktion gibt zufällig entweder 1 oder -1 zurück
function signum = random_sign()
    r = randi([0, 1]);
    signum = 2*r - 1;
end

% Diese Funktion erzeugt einen zufälligen Parameter (float) im gegebenen
% Intervall
function r_param = random_param(lower_barrier, upper_barrier)
    r_param = lower_barrier + (upper_barrier - lower_barrier) * rand();
end

% Diese Funktion erzeugt zufällige Anfangsbedingungen im gegebenen
% Intervall
function x_0 = random_init(n, lower_barrier, upper_barrier, random_signum)
    % leeren Vektor
    x_0 = zeros(n, 1);
    
    % Ohne zufälliges Vorzeichen
    if random_signum == false
        for i = 1:n
            x_0(i) = random_param(lower_barrier, upper_barrier);
        end
    end

    % Mit zufälligem Vorzeichen
    if random_signum == true
        for i = 1:n
            x_0(i) = random_sign()*random_param(lower_barrier, upper_barrier);
        end
    end
end

% Diese Funktion wertet die inverse Dynamik eines einfachen 1 FHG Systems
% aus, um gute Drehmomentverläufe zu generieren. Diese Drehmomentverläufe
% werden später zur Simulation verwendet.
function tau = inv_dyn_1_FHG(traj_data)

    % Systemparameter
    J_kgm2 = 0.06;  % hier etwa J1 + J2, damit der Motor beide beschleunigt

    % Auswertung inverse Dynamik
    tau = J_kgm2.*traj_data.q1_pp;

end

% Dies ist die Zustandsraumdarstellung des Torsionsschwingers
function x_p = ZRD_2_FHG(t, x, tau_vec, t_vec, J_1_kgm2, J_2_kgm2, c_phi_Nmprad)

    % Drehmoment interpolieren
    tau = interp1(t_vec, tau_vec, t);

    % Vektoren auslesem
    q = [x(1);
        x(2)];
    q_p = [x(3);
        x(4)];

    % Systemdifferentialgleichungen auswerten
    M = [J_1_kgm2 0;
        0 J_2_kgm2];

    K = [c_phi_Nmprad -c_phi_Nmprad;
        -c_phi_Nmprad c_phi_Nmprad];

    q_pp = inv(M)*([tau; 0] - K*q);
    
    % Ableitung Zustandsvektor berechnen
    x_p = [q_p;
        q_pp];

end

%% Parameterdefinition

% Bewegungszeit und Schrittweite
samples_per_run = 100;
move_time = 1; 
t_vec = linspace(0, move_time, samples_per_run);

% Systemparameter
J_1_kgm2 = 0.029;
J_2_kgm2 = 0.029;
c_phi_Nmprad = 7.309;

% Zeitpunkte der Wegpunkte
waypointTimes = [0 move_time];

% Anzahl der voneinander unabhängigen Bewegungen
number_runs = 12;

% Seed für reproduzierbare Ergebnisse
rng(42)

% Sollen Simulationsdaten gespeichert werden
savedata = true;

% Sollen plots gespeichert werden
save_plots = true;

%% Wegpunkte für Trajektorie festlegen (hier unterscheiden bei Robotermodellen)

% Startpunkte
q1_0 = random_init(number_runs, 0, 2*pi, false); % Bsp. Intervall r [0, 0.5]

% Differenzen für Endpunkte
delta_q1 = random_init(number_runs, 2*pi, 4*pi, true); % Bsp. Intervall delta_r [0.2, 0.5]

%% Trajektorien generieren

% Struct zur speicherung der Daten vorbereiten
traj_data(number_runs) = struct();

for i = 1:number_runs

    % Wegpunkte
    waypoints_q1 = [q1_0(i), (q1_0(i) + delta_q1(i))];

    % Berechnung der Trajektorien
    [q1, q1_p, q1_pp] = quinticpolytraj(waypoints_q1, waypointTimes, t_vec);

    % Trajektorien speichern
    traj_data(i).q1 = q1';
    traj_data(i).q1_p = q1_p';
    traj_data(i).q1_pp = q1_pp';

end

%% Drehmomente generieren (1 FHG System - nur Beschleunigungsterme)

for i = 1:number_runs

    % inverse Dynamik auswerten
    traj_data(i).tau = inv_dyn_1_FHG(traj_data(i));

end

%% Simulation des Systems durchführen

% Struct zur speicherung der simulierten Daten vorbereiten
sim_data(number_runs) = struct();

for i = 1:number_runs
    
    % Anfangsbedingungen
    x_0 = [traj_data(i).q1(1);
        traj_data(i).q1(1);
        traj_data(i).q1_p(1);
        traj_data(i).q1_p(1)];

    % Stellgrößen
    tau_vec = traj_data(i).tau;

    % ODE-Funktion mit Parametern
    odefun = @(t, x) ZRD_2_FHG(t, x, tau_vec, t_vec, J_1_kgm2, J_2_kgm2, c_phi_Nmprad);

    % Solver zur Lösung der DGL
    options = odeset('Stats', 'on');
    [t, x] = ode45(odefun, t_vec, x_0, options);

    % Beschleunigungen rekonsruieren
    x_p = zeros(size(x));
    g = zeros([samples_per_run, 2]);

    for j = 1:length(t_vec)
        
        % Zusatndsvektor extrahieren
        x_j = x(j, :);

        % ZRD auswerten
        x_p_j = ZRD_2_FHG(t_vec(j), x_j, tau_vec, t_vec, J_1_kgm2, J_2_kgm2, c_phi_Nmprad);

        % Federkräfte Berechnen
        K = [c_phi_Nmprad -c_phi_Nmprad;
            -c_phi_Nmprad c_phi_Nmprad];

        g_j = K*[x_j(1); x_j(2)];

        % Speichern
        x_p(j, :) = x_p_j;
        g(j, :) = g_j;

    end

    % Speichern
    sim_data(i).q1 = x(:, 1);
    sim_data(i).q2 = x(:, 2);
    sim_data(i).q1_p = x(:, 3);
    sim_data(i).q2_p = x(:, 4);
    sim_data(i).q1_pp = x_p(:, 3);
    sim_data(i).q2_pp = x_p(:, 4);

    sim_data(i).tau_1 = tau_vec;
    sim_data(i).tau_2 = zeros(size(tau_vec));

    sim_data(i).M_11 = ones(size(tau_vec)) * J_1_kgm2;
    sim_data(i).M_12 = zeros(size(tau_vec));
    sim_data(i).M_22 = ones(size(tau_vec)) * J_2_kgm2;

    sim_data(i).C_1 = zeros(size(tau_vec));
    sim_data(i).C_2 = zeros(size(tau_vec));

    sim_data(i).g_1 = g(:, 1);
    sim_data(i).g_2 = g(:, 2);

end

%% Simulationsergebnisse Speichern (Daten in Trainings- und Testdaten aufteilen)

% Testdaten auswählen und in Features und Labels aufteilen (20%)
number_testdata = floor((number_runs)/5);
test_idx = randperm((number_runs), number_testdata);   % Zufällige Indizees für Testdaten

features_test = [sim_data(test_idx(1)).q1, sim_data(test_idx(1)).q2, sim_data(test_idx(1)).q1_p, sim_data(test_idx(1)).q2_p, sim_data(test_idx(1)).tau_1, sim_data(test_idx(1)).tau_2];
labels_test = [sim_data(test_idx(1)).q1_pp, sim_data(test_idx(1)).q2_pp];
Mass_Cor_test = [sim_data(test_idx(1)).M_11, sim_data(test_idx(1)).M_12, sim_data(test_idx(1)).M_22, sim_data(test_idx(1)).C_1, sim_data(test_idx(1)).C_2, sim_data(test_idx(1)).g_1, sim_data(test_idx(1)).g_2];

for i = test_idx(2:end)
    % Features [q1, q2, q1_p, q2_p, tau_1, tau_2]
    features_test = [features_test;
        sim_data(i).q1, sim_data(i).q2, sim_data(i).q1_p, sim_data(i).q2_p, sim_data(i).tau_1, sim_data(i).tau_2];
    % Labels [q1_pp, q2_pp]
    labels_test = [labels_test;
        sim_data(i).q1_pp, sim_data(i).q2_pp];
    % Mass and Coriolis Terms [M_11, M_12, M_22, C_1, C_2, g_1, g_2]
    Mass_Cor_test = [Mass_Cor_test;
        sim_data(i).M_11, sim_data(i).M_12, sim_data(i).M_22, sim_data(i).C_1, sim_data(i).C_2, sim_data(i).g_1, sim_data(i).g_2];
end

% Trainingsdaten auswählen und in Features und Labels aufteilen (80%)
training_idx = setdiff((1:number_runs), test_idx);

features_training = [sim_data(training_idx(1)).q1, sim_data(training_idx(1)).q2, sim_data(training_idx(1)).q1_p, sim_data(training_idx(1)).q2_p, sim_data(training_idx(1)).tau_1, sim_data(training_idx(1)).tau_2];
labels_training = [sim_data(training_idx(1)).q1_pp, sim_data(training_idx(1)).q2_pp];
Mass_Cor_training = [sim_data(training_idx(1)).M_11, sim_data(training_idx(1)).M_12, sim_data(training_idx(1)).M_22, sim_data(training_idx(1)).C_1, sim_data(training_idx(1)).C_2, sim_data(training_idx(1)).g_1, sim_data(training_idx(1)).g_2];

for i = training_idx(2:end)
    % Features [q1, q2, q1_p, q2_p, tau_1, tau_2]
    features_training = [features_training;
        sim_data(i).q1, sim_data(i).q2, sim_data(i).q1_p, sim_data(i).q2_p, sim_data(i).tau_1, sim_data(i).tau_2];
    % Labels [q1_pp, q2_pp]
    labels_training = [labels_training;
        sim_data(i).q1_pp, sim_data(i).q2_pp];
    % Mass and Coriolis Terms [M_11, M_12, M_22, C_1, C_2, g_1, g_2]
    Mass_Cor_training = [Mass_Cor_training;
        sim_data(i).M_11, sim_data(i).M_12, sim_data(i).M_22, sim_data(i).C_1, sim_data(i).C_2,sim_data(i).g_1, sim_data(i).g_2];
end

if savedata == true
    % Pfad dieses Skripts
    my_path = fileparts(mfilename('fullpath'));

    % Zielordner relativ zum Skriptpfad
    target_folder = fullfile(my_path, '..', 'Training_Data', 'MATLAB_Simulation');
    target_folder = fullfile(target_folder); % Pfad normalisieren

    % Datei speichern
    num_samples = num2str(length(features_test) + length(features_training));
    time_stamp = string(datetime('now', 'Format', 'yyyy_MM_dd_HH_mm_ss'));
    dateiName = 'SimData_Torsionsschwinger_' + time_stamp + '_Samples_' + num_samples + '.mat';   
    full_path = fullfile(target_folder, dateiName);
    save(full_path, 'features_training', 'labels_training', 'features_test', 'labels_test', "Mass_Cor_test");
end

%% Plots + Speichern

% Speicher Pfad
plot_path = 'D:\Programmierung_Ole\Latex\Studienarbeit_Repo_Overleaf\Bilder\04_Datengenerierung';
plot_1_name = fullfile(plot_path, 'Abbildung_phi_phip_phipp_Torsionsschwinger.pdf');

% Zeitvektor über alle Trainingstrajektorien
t_vec_ges = linspace(0, (number_runs - number_testdata) * move_time, (number_runs - number_testdata) * samples_per_run)';

% Plot 2 Geschwindigkeit Beschleunigung
Triple_Subplot(t_vec_ges, {[features_training(:, 1), features_training(:, 2)], [features_training(:, 3), features_training(:, 4)], [labels_training(:, 1), labels_training(:, 2)]}, ...
    '$\mathrm{Zeit} \, / \, \mathrm{s}$', ...
    {'$\mathrm{Pos.} \, / \, \mathrm{rad}$', '$\mathrm{Geschw.} \, / \, \mathrm{rad} \, \mathrm{s}^{-1}$', '$\mathrm{Beschl.} \, / \, \mathrm{rad} \, \mathrm{s}^{-2}$'}, ...
    {'', '', ''}, ...
    {{'$\varphi_{T, 1}$', '$\varphi_{T, 2}$'}, {'$\dot{\varphi}_{T, 1}$', '$\dot{\varphi}_{T, 2}$'}, {'$\ddot{\varphi}_{T, 1}$', '$\ddot{\varphi}_{T, 2}$'}}, ...
    plot_1_name, save_plots, true)

%% Ergebnisse Plotten (nur Trainingsdaten)

% Plot q1, q2
figure('WindowState','maximized');

subplot(2,3,1);
plot(t_vec_ges, features_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q1');
grid on;
title('q1(t)');

subplot(2,3,4);
plot(t_vec_ges, features_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q2');
grid on;
title('q2(t)');

% Plot q1_p, q2_p
subplot(2,3,2);
plot(t_vec_ges, features_training(:, 3), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q1_p');
grid on;
title('q1p(t)');

subplot(2,3,5);
plot(t_vec_ges, features_training(:, 4), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q2_p');
grid on;
title('q2p(t)');

% Plot q1_pp, q2_pp
subplot(2,3,3);
plot(t_vec_ges, labels_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q1pp');
grid on;
title('q1pp(t)');

subplot(2,3,6);
plot(t_vec_ges, labels_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('q2pp');
grid on;
title('q2pp(t)');

% Plot F, tau
figure('WindowState','maximized');

subplot(2,1,1);
plot(t_vec_ges, features_training(:, 5), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('tau_1');
grid on;
title('tau1(t)');

subplot(2,1,2);
plot(t_vec_ges, features_training(:, 6), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('tau_2');
grid on;
title('tau2(t)');

% Plott M_11, M_12, M_22
figure('WindowState','maximized');

subplot(2,4,1);
plot(t_vec_ges, Mass_Cor_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M11');
grid on;
title('M11(t)');

subplot(2,4,5);
plot(t_vec_ges, Mass_Cor_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M12');
grid on;
title('M12(t)');

subplot(2,4,2);
plot(t_vec_ges, Mass_Cor_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M12');
grid on;
title('M12(t)');

subplot(2,4,6);
plot(t_vec_ges, Mass_Cor_training(:, 3), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M22');
grid on;
title('M22(t)');

% Plott C_1, C_2
subplot(2,4,3);
plot(t_vec_ges, Mass_Cor_training(:, 4), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('C1');
grid on;
title('C1(t)');

subplot(2,4,7);
plot(t_vec_ges, Mass_Cor_training(:, 5), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('C2');
grid on;
title('C2(t)');

% Plott g_1, g_2
subplot(2,4,4);
plot(t_vec_ges, Mass_Cor_training(:, 6), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('g1');
grid on;
title('g1(t)');

subplot(2,4,8);
plot(t_vec_ges, Mass_Cor_training(:, 7), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('g2');
grid on;
title('g2(t)');

