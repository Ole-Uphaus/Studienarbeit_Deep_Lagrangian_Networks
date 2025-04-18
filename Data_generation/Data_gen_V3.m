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
number_runs = 20;

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

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

%% Simulationsergebnisse Speichern (Daten in Trainings- und Testdaten aufteilen)

% Testdaten auswählen und in Features und Labels aufteilen (20%)
number_testdata = floor((number_runs)/5);
test_idx = randperm((number_runs), number_testdata);   % Zufällige Indizees für Testdaten

features_test = [traj_data(test_idx(1)).r, traj_data(test_idx(1)).phi, traj_data(test_idx(1)).r_p, traj_data(test_idx(1)).phi_p];
labels_test = [traj_data(test_idx(1)).r_pp, traj_data(test_idx(1)).phi_pp];

for i = test_idx(2:end)
    % Features [r, phi, r_p, phi_p, F, tau]
    features_test = [features_test;
        traj_data(i).r, traj_data(i).phi, traj_data(i).r_p, traj_data(i).phi_p];
    % Labels [r_pp, phi_pp]
    labels_test = [labels_test;
        traj_data(i).r_pp, traj_data(i).phi_pp];
end

% Trainingsdaten auswählen und in Features und Labels aufteilen (80%)
training_idx = setdiff((1:number_runs), test_idx);

features_training = [traj_data(training_idx(1)).r, traj_data(training_idx(1)).phi, traj_data(training_idx(1)).r_p, traj_data(training_idx(1)).phi_p];
labels_training = [traj_data(training_idx(1)).r_pp, traj_data(training_idx(1)).phi_pp];

for i = training_idx(2:end)
    % Features [r, phi, r_p, phi_p, F, tau]
    features_training = [features_training;
        traj_data(i).r, traj_data(i).phi, traj_data(i).r_p, traj_data(i).phi_p];
    % Labels [r_pp, phi_pp]
    labels_training = [labels_training;
        traj_data(i).r_pp, traj_data(i).phi_pp];
end

%% Ergebnisse Plotten (nur Trainingsdaten)

% Zeitvektor über alle Trainingstrajektorien
t_vec_ges = linspace(0, (number_runs - number_testdata) * move_time, (number_runs - number_testdata) * smples_per_run)';

% Plot r, phi
figure();

subplot(2,1,1);
plot(t_vec_ges, features_training(:, 1), 'b', 'LineWidth', 1.5);

xlabel('Zeit [s]');
ylabel('Weg [m]');
grid on;
title('Position r(t)');

subplot(2,1,2);
plot(t_vec_ges, features_training(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi [rad]');
grid on;
title('Winkel phi(t)');

