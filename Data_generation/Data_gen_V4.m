% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      17.04.2025
% Beschreibung:
% Dies ist eine ähnliche Version wie Data_gen_V3. Ich werde jedoch in
% diesem Skript andere Trajektorien generieren. Anstatt wie in V3 einfach
% immer kleine Übergänge zu berechnen, werde ich jetzt Pendelbewegungen des
% Roboters durchführen und dazu cosinus Funktionen verwenden. Es ändert
% sich also nur die Generierung der Trajektorien. Die Auswertung der
% Inversen Dynamik bleibt estehen.
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

% Diese funktion wertet bei gegebenen Trajektorien (q, q_p, q_pp) die
% Inverse Dynamik des Robotermodells aus (Ursprüngliches Modell)
function out = inv_dyn_2_FHG_Robot_1(traj_data)

    % Systemparameter
    m_kg = 5;   % Masse des Arms
    mL_kg = 2;  % Masse der Last
    J_kgm2 = 0.4;  % gesamte Rotationsträgheit
    l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

    % Output definieren
    out = struct();

    % Daten aus Trajektoren extrahieren
    r = traj_data.q1;
    r_p = traj_data.q1_p;
    r_pp = traj_data.q1_pp;

    phi = traj_data.q2;
    phi_p = traj_data.q2_p;
    phi_pp = traj_data.q2_pp;

    % Massenmatrix
    out.M_11 = m_kg + mL_kg + r*0;    % Oberer linker Eintrag der Massenmatrix (r*0 damit Vektor herauskommt)
    out.M_12 = r*0; % Oberer Rechter Eintrag der Massenmatrix (r*0 damit Vektor herauskommt)
    out.M_22 = J_kgm2 + m_kg*(r - l_m).^2 + mL_kg*r.^2;    % Unterer rechter Eintrag der Massenmatrix

    % Corioliskräfte
    out.C_1 = -(mL_kg*r + m_kg*(r - l_m)).*phi_p.^2; % Erster Vektoreintrag
    out.C_2 = 2*(m_kg*(r - l_m) + mL_kg*r).*r_p.*phi_p; % Zweiter Vektoreintrag

    % Gewichtskräfte
    out.g_1 = r*0;  % Erster Vektoreintrag
    out.g_2 = r*0;  % Zweiter Vektoreintrag

    % Eingeprägte Kräfte/Momente (Hier direkt Massen- und Coriolisterme
    % eingesetzt)
    out.tau_1 = out.M_11.*r_pp + out.C_1;   % Eigentlich F
    out.tau_2 = out.M_22.*phi_pp + out.C_2; % Eigentlich tau

end

% Diese funktion wertet bei gegebenen Trajektorien (q, q_p, q_pp) die
% Inverse Dynamik des Robotermodells aus (Neues Modell)
function out = inv_dyn_2_FHG_Robot_2(traj_data)

    % Systemparameter
    m1_kg = 1;  % Masse Arm 1
    m2_kg = 0.5;  % Masse Arm 2
    l1_m = 0.3; % Länge Arm 1
    l2_m = 0.3; % Länge Arm 2
    J1_kgm2 = 1;  % Rotationsträgheit Arm 1
    J2_kgm2 = 1;  % Rotationsträgheit Arm 2
    g_mps2 = 9.81;  % Gravitationskonstante

    % Output definieren
    out = struct();

    % Daten aus Trajektoren extrahieren
    phi_1 = traj_data.q1;
    phi_1_p = traj_data.q1_p;
    phi_1_pp = traj_data.q1_pp;

    phi_2 = traj_data.q2;
    phi_2_p = traj_data.q2_p;
    phi_2_pp = traj_data.q2_pp;

    % Massenmatrix
    out.M_11 = J1_kgm2 + J2_kgm2 + (l1_m^2*m1_kg)/4 + l1_m^2*m2_kg + (l2_m^2*m2_kg)/4 + l1_m*l2_m*m2_kg.*cos(phi_2);   % Oberer linker Eintrag der Massenmatrix
    out.M_12 = (m2_kg*l2_m^2)/4 + (l1_m*m2_kg.*cos(phi_2).*l2_m)/2 + J2_kgm2;      % Oberer Rechter Eintrag der Massenmatrix
    out.M_22 = (m2_kg*l2_m^2)/4 + J2_kgm2 + phi_1*0;    % Unterer rechter Eintrag der Massenmatrix

    % Corioliskräfte
    out.C_1 = -(l1_m*l2_m*m2_kg.*phi_2_p.*sin(phi_2).*(2.*phi_1_p + phi_2_p))/2;     % Erster Vektoreintrag
    out.C_2 = (l1_m*l2_m*m2_kg.*phi_1_p.^2.*sin(phi_2))/2;    % Zweiter Vektoreintrag

    % Gewichtskräfte
    out.g_1 = g_mps2*m2_kg.*((l2_m.*sin(phi_1 + phi_2))/2 + l1_m.*sin(phi_1)) + (g_mps2*l1_m*m1_kg.*sin(phi_1))/2;  % Erster Vektoreintrag
    out.g_2 = (g_mps2*l2_m*m2_kg.*sin(phi_1 + phi_2))/2;  % Zweiter Vektoreintrag

    % Eingeprägte Kräfte/Momente (Hier direkt Massen- und Coriolisterme
    % eingesetzt)
    out.tau_1 = out.M_11.*phi_1_pp + out.M_12.*phi_2_pp + out.C_1 + out.g_1;    % tau_1
    out.tau_2 = out.M_22.*phi_2_pp + out.M_12.*phi_1_pp + out.C_2 + out.g_2;    % tau_2

end

%% Parameterdefinition

% Bewegungszeit und Schrittweite
samples_per_run = 100;
move_time = 2;  % Vrogeschlagene Werte: Rob_Model = 1 (3s), Rob_Model = 2 (s)
t_vec = linspace(0, move_time, samples_per_run);

% Zeitpunkte der Wegpunkte
waypointTimes = [0 move_time];

% Anzahl der voneinander unabhängigen Bewegungen
number_runs = 10;

% Robotermodell auswählen (1 - Roboter aus NLRS, 2 - Roboter mit 2
% Drehgelenken)
Rob_Model = 2;

% Seed für reproduzierbare Ergebnisse
rng(42)

% Sollen Simulationsdaten gespeichert werden
savedata = false;

%% Wegpunkte für Trajektorie festlegen (hier unterscheiden bei Robotermodellen)

if Rob_Model == 1
    % Startpunkte
    q1_0 = random_init(number_runs, 0, 0.5, false); % Bsp. Intervall r [0, 0.5]
    q2_0 = random_init(number_runs, 0, pi, false);  % Bsp. Intervall phi [0, pi]
    
    % Differenzen für Endpunkte
    delta_q1 = random_init(number_runs, 0.2, 0.5, true); % Bsp. Intervall delta_r [0.2, 0.5]
    delta_q2 = random_init(number_runs, 1/4*pi, 3/4*pi, true); % Bsp. Intervall delta_phi [1/4*pi, 3/4*pi]

elseif Rob_Model == 2
    % Startpunkte
    q1_0 = random_init(number_runs, 1/4*pi, 1/2*pi, true); % Bsp. Intervall phi_1 
    q2_0 = random_init(number_runs, 0, 1/8*pi, true);  % Bsp. Intervall phi_2 
    
    % Differenzen für Endpunkte
    delta_q1 = random_init(number_runs, 1/3*pi, 1/2*pi, true); % Bsp. Intervall delta_phi_1 
    delta_q2 = random_init(number_runs, 1/3*pi, 1/2*pi, true); % Bsp. Intervall delta_phi_2 

end

%% Trajektorien generieren

% Struct zur speicherung der Daten vorbereiten
traj_data(number_runs) = struct();

for i = 1:number_runs

    % Wegpunkte
    waypoints_q1 = [q1_0(i), (q1_0(i) + delta_q1(i))];
    waypoints_q2 = [q2_0(i), (q2_0(i) + delta_q2(i))];

    % Berechnung der Trajektorien
    [q1, q1_p, q1_pp] = quinticpolytraj(waypoints_q1, waypointTimes, t_vec);
    [q2, q2_p, q2_pp] = quinticpolytraj(waypoints_q2, waypointTimes, t_vec);

    % Trajektorien speichern
    traj_data(i).q1 = q1';
    traj_data(i).q1_p = q1_p';
    traj_data(i).q1_pp = q1_pp';

    traj_data(i).q2 = q2';
    traj_data(i).q2_p = q2_p';
    traj_data(i).q2_pp = q2_pp';

end

%% Inverse Dynamik auswerten + Massen- und Coriolisterme berechnen (Differentialgleichung)

for i = 1:number_runs

    % Ausgewähltes Modell nutzen
    if Rob_Model == 1
        out = inv_dyn_2_FHG_Robot_1(traj_data(i));
    elseif Rob_Model == 2
        out = inv_dyn_2_FHG_Robot_2(traj_data(i));
    end

    % Massenmatrix
    traj_data(i).M_11 = out.M_11;
    traj_data(i).M_12 = out.M_12;
    traj_data(i).M_22 = out.M_22;

    % Corioliskräfte
    traj_data(i).C_1 = out.C_1;
    traj_data(i).C_2 = out.C_2;

    % Gewichtskräfte
    traj_data(i).g_1 = out.g_1;
    traj_data(i).g_2 = out.g_2;

    % Eingeprägte Kräfte/Momente 
    traj_data(i).tau_1 = out.tau_1;
    traj_data(i).tau_2 = out.tau_2;

end

%% Simulationsergebnisse Speichern (Daten in Trainings- und Testdaten aufteilen)

% Testdaten auswählen und in Features und Labels aufteilen (20%)
number_testdata = floor((number_runs)/5);
test_idx = randperm((number_runs), number_testdata);   % Zufällige Indizees für Testdaten

features_test = [traj_data(test_idx(1)).q1, traj_data(test_idx(1)).q2, traj_data(test_idx(1)).q1_p, traj_data(test_idx(1)).q2_p, traj_data(test_idx(1)).tau_1, traj_data(test_idx(1)).tau_2];
labels_test = [traj_data(test_idx(1)).q1_pp, traj_data(test_idx(1)).q2_pp];
Mass_Cor_test = [traj_data(test_idx(1)).M_11, traj_data(test_idx(1)).M_12, traj_data(test_idx(1)).M_22, traj_data(test_idx(1)).C_1, traj_data(test_idx(1)).C_2, traj_data(test_idx(1)).g_1, traj_data(test_idx(1)).g_2];

for i = test_idx(2:end)
    % Features [q1, q2, q1_p, q2_p, tau_1, tau_2]
    features_test = [features_test;
        traj_data(i).q1, traj_data(i).q2, traj_data(i).q1_p, traj_data(i).q2_p, traj_data(i).tau_1, traj_data(i).tau_2];
    % Labels [q1_pp, q2_pp]
    labels_test = [labels_test;
        traj_data(i).q1_pp, traj_data(i).q2_pp];
    % Mass and Coriolis Terms [M_11, M_12, M_22, C_1, C_2, g_1, g_2]
    Mass_Cor_test = [Mass_Cor_test;
        traj_data(i).M_11, traj_data(i).M_12, traj_data(i).M_22, traj_data(i).C_1, traj_data(i).C_2, traj_data(i).g_1, traj_data(i).g_2];
end

% Trainingsdaten auswählen und in Features und Labels aufteilen (80%)
training_idx = setdiff((1:number_runs), test_idx);

features_training = [traj_data(training_idx(1)).q1, traj_data(training_idx(1)).q2, traj_data(training_idx(1)).q1_p, traj_data(training_idx(1)).q2_p, traj_data(training_idx(1)).tau_1, traj_data(training_idx(1)).tau_2];
labels_training = [traj_data(training_idx(1)).q1_pp, traj_data(training_idx(1)).q2_pp];
Mass_Cor_training = [traj_data(training_idx(1)).M_11, traj_data(training_idx(1)).M_12, traj_data(training_idx(1)).M_22, traj_data(training_idx(1)).C_1, traj_data(training_idx(1)).C_2, traj_data(training_idx(1)).g_1, traj_data(training_idx(1)).g_2];

for i = training_idx(2:end)
    % Features [q1, q2, q1_p, q2_p, tau_1, tau_2]
    features_training = [features_training;
        traj_data(i).q1, traj_data(i).q2, traj_data(i).q1_p, traj_data(i).q2_p, traj_data(i).tau_1, traj_data(i).tau_2];
    % Labels [q1_pp, q2_pp]
    labels_training = [labels_training;
        traj_data(i).q1_pp, traj_data(i).q2_pp];
    % Mass and Coriolis Terms [M_11, M_12, M_22, C_1, C_2, g_1, g_2]
    Mass_Cor_training = [Mass_Cor_training;
        traj_data(i).M_11, traj_data(i).M_12, traj_data(i).M_22, traj_data(i).C_1, traj_data(i).C_2,traj_data(i).g_1, traj_data(i).g_2];
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
    dateiName = 'SimData_V3_Rob_Model_' + string(Rob_Model) + '_' + time_stamp + '_Samples_' + num_samples + '.mat';
    full_path = fullfile(target_folder, dateiName);
    save(full_path, 'features_training', 'labels_training', 'features_test', 'labels_test', "Mass_Cor_test");
end

%% Ergebnisse Plotten (nur Trainingsdaten)

% Zeitvektor über alle Trainingstrajektorien
t_vec_ges = linspace(0, (number_runs - number_testdata) * move_time, (number_runs - number_testdata) * samples_per_run)';

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

