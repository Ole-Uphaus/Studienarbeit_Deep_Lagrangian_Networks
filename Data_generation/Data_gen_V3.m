% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      17.04.2025
% Beschreibung:
% Dies ist eine komplett neue Version des Skriptes zur Generierung der
% Trainingsdaten. Ich werde in diesem skript nicht so viel dem Zufall
% überlassen und Solltrajektorien für r und phi vorgeben (mit einem
% gewissen Zufallsanteil). Aus diesen Solltrajektorien werde ich
% anschließend die Ableitungen berechnen und die inverse Dynamik auswerten.
% Zur Generierung der Trainingsdaten muss somit keine ODE gelöst werden.
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

%% Parameterdefinition

% Bewegungszeit und Schrittweite
smples_per_run = 100;
move_time = 5;
t_vec = linspace(0, move_time, smples_per_run);

% Zeitpunkte der Wegpunkte
waypointTimes = [0 move_time];

% Anzahl der voneinander unabhängigen Bewegungen
number_runs = 15;

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

% Sollen Simulationsdaten gespeichert werden
savedata = true;

%% Wegpunkte für Trajektorie festlegen

% Startpunkte
r_0 = random_init(number_runs, 0, 0.5, false); % Intervall [0, 0.5]
phi_0 = random_init(number_runs, 0, pi, false);  % Intervall [0, pi]

% Differenzen für Endpunkte
delta_r = random_init(number_runs, 0.2, 0.5, true); % Intervall [0.2, 0.5]
delta_phi = random_init(number_runs, 1/4*pi, 3/4*pi, true); % Intervall [1/4*pi, 3/4*pi]

%% Trajektorien generieren

% Struct zur speicherung der Daten vorbereiten
traj_data(number_runs) = struct();

for i = 1:number_runs

    % Wegpunkte
    waypoints_r = [r_0(i), (r_0(i) + delta_r(i))];
    waypoints_phi = [phi_0(i), (phi_0(i) + delta_phi(i))];

    % Berechnung der Trajektorien
    [r, r_p, r_pp] = quinticpolytraj(waypoints_r, waypointTimes, t_vec);
    [phi, phi_p, phi_pp] = quinticpolytraj(waypoints_phi, waypointTimes, t_vec);

    % Trajektorien speichern
    traj_data(i).r = r';
    traj_data(i).r_p = r_p';
    traj_data(i).r_pp = r_pp';

    traj_data(i).phi = phi';
    traj_data(i).phi_p = phi_p';
    traj_data(i).phi_pp = phi_pp';

end

%% Inverse Dynamik auswerten + Massen- und Coriolisterme berechnen (Differentialgleichung)

for i = 1:number_runs

    % Daten aus Trajektoren extrahieren
    r = traj_data(i).r;
    r_p = traj_data(i).r_p;
    r_pp = traj_data(i).r_pp;

    phi = traj_data(i).phi;
    phi_p = traj_data(i).phi_p;
    phi_pp = traj_data(i).phi_pp;

    % Massenmatrix
    traj_data(i).M_11 = m_kg + mL_kg + r*0;    % Oberer linker Eintrag der Massenmatrix (r*0 damit Vektor herauskommt)
    traj_data(i).M_22 = J_kgm2 + m_kg*(r - l_m).^2 + mL_kg*r.^2;    % Unterer rechter Eintrag der Massenmatrix

    % Corioliskräfte
    traj_data(i).C_1 = -(mL_kg*r + m_kg*(r - l_m)).*phi_p.^2; % Erster Vektoreintrag
    traj_data(i).C_2 = 2*(m_kg*(r - l_m) + mL_kg*r).*r_p.*phi_p; % Zweiter Vektoreintrag

    % Eingeprägte Kräfte/Momente (Hier direkt Massen- und Coriolisterme
    % eingesetzt)
    traj_data(i).F = traj_data(i).M_11.*r_pp + traj_data(i).C_1;
    traj_data(i).tau = traj_data(i).M_22.*phi_pp + traj_data(i).C_2;

end

%% Simulationsergebnisse Speichern (Daten in Trainings- und Testdaten aufteilen)

% Testdaten auswählen und in Features und Labels aufteilen (20%)
number_testdata = floor((number_runs)/5);
test_idx = randperm((number_runs), number_testdata);   % Zufällige Indizees für Testdaten

features_test = [traj_data(test_idx(1)).r, traj_data(test_idx(1)).phi, traj_data(test_idx(1)).r_p, traj_data(test_idx(1)).phi_p, traj_data(test_idx(1)).F, traj_data(test_idx(1)).tau];
labels_test = [traj_data(test_idx(1)).r_pp, traj_data(test_idx(1)).phi_pp];
Mass_Cor_test = [traj_data(test_idx(1)).M_11, traj_data(test_idx(1)).M_22, traj_data(test_idx(1)).C_1, traj_data(test_idx(1)).C_2];

for i = test_idx(2:end)
    % Features [r, phi, r_p, phi_p, F, tau]
    features_test = [features_test;
        traj_data(i).r, traj_data(i).phi, traj_data(i).r_p, traj_data(i).phi_p, traj_data(i).F, traj_data(i).tau];
    % Labels [r_pp, phi_pp]
    labels_test = [labels_test;
        traj_data(i).r_pp, traj_data(i).phi_pp];
    % Mass and Coriolis Terms [M_11, M_22, C_1, C_2]
    Mass_Cor_test = [Mass_Cor_test;
        traj_data(i).M_11, traj_data(i).M_22, traj_data(i).C_1, traj_data(i).C_2];
end

% Trainingsdaten auswählen und in Features und Labels aufteilen (80%)
training_idx = setdiff((1:number_runs), test_idx);

features_training = [traj_data(training_idx(1)).r, traj_data(training_idx(1)).phi, traj_data(training_idx(1)).r_p, traj_data(training_idx(1)).phi_p, traj_data(training_idx(1)).F, traj_data(training_idx(1)).tau];
labels_training = [traj_data(training_idx(1)).r_pp, traj_data(training_idx(1)).phi_pp];
Mass_Cor_training = [traj_data(training_idx(1)).M_11, traj_data(training_idx(1)).M_22, traj_data(training_idx(1)).C_1, traj_data(training_idx(1)).C_2];

for i = training_idx(2:end)
    % Features [r, phi, r_p, phi_p, F, tau]
    features_training = [features_training;
        traj_data(i).r, traj_data(i).phi, traj_data(i).r_p, traj_data(i).phi_p, traj_data(i).F, traj_data(i).tau];
    % Labels [r_pp, phi_pp]
    labels_training = [labels_training;
        traj_data(i).r_pp, traj_data(i).phi_pp];
    % Mass and Coriolis Terms [M_11, M_22, C_1, C_2]
    Mass_Cor_training = [Mass_Cor_training;
        traj_data(i).M_11, traj_data(i).M_22, traj_data(i).C_1, traj_data(i).C_2];
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
    dateiName = 'SimData_V3_' + time_stamp + '_Samples_' + num_samples + '.mat';
    full_path = fullfile(target_folder, dateiName);
    save(full_path, 'features_training', 'labels_training', 'features_test', 'labels_test', "Mass_Cor_test");
end

%% Ergebnisse Plotten (nur Trainingsdaten)

% Zeitvektor über alle Trainingstrajektorien
t_vec_ges = linspace(0, (number_runs - number_testdata) * move_time, (number_runs - number_testdata) * smples_per_run)';

% Plot r, phi
figure('WindowState','maximized');

subplot(2,3,1);
plot(t_vec_ges, features_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('r [m]');
grid on;
title('Position r(t)');

subplot(2,3,4);
plot(t_vec_ges, features_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi [rad]');
grid on;
title('Winkel phi(t)');

% Plot r_p, phi_p
subplot(2,3,2);
plot(t_vec_ges, features_training(:, 3), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('rp [m/s]');
grid on;
title('Geschwindigkeit rp(t)');

subplot(2,3,5);
plot(t_vec_ges, features_training(:, 4), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phip [rad]');
grid on;
title('Winkelgeschwindigkeit phip(t)');

% Plot r_pp, phi_pp
subplot(2,3,3);
plot(t_vec_ges, labels_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('rpp [m/s]');
grid on;
title('Beschleunigung rpp(t)');

subplot(2,3,6);
plot(t_vec_ges, labels_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phipp [rad]');
grid on;
title('Winkelbeschleunigung phipp(t)');

% Plot F, tau
figure('WindowState','maximized');

subplot(2,3,1);
plot(t_vec_ges, features_training(:, 5), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('F [N]');
grid on;
title('Kraft F(t)');

subplot(2,3,4);
plot(t_vec_ges, features_training(:, 6), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('T [Nm]');
grid on;
title('Drehmoment T(t)');

% Plott M_11, M_22
subplot(2,3,2);
plot(t_vec_ges, Mass_Cor_training(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M11 [kg]');
grid on;
title('Massenmatrix M11(t)');

subplot(2,3,5);
plot(t_vec_ges, Mass_Cor_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('M22 [kgm2]');
grid on;
title('Massenmatrix M22(t)');

% Plott C_1, C_2
subplot(2,3,3);
plot(t_vec_ges, Mass_Cor_training(:, 3), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('C1 [N]');
grid on;
title('Corioliskraft C1(t)');

subplot(2,3,6);
plot(t_vec_ges, Mass_Cor_training(:, 4), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('C2 [N]');
grid on;
title('Corioliskraft C2(t)');
