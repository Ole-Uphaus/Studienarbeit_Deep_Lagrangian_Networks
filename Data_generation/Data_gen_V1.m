% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      18.03.2025
% Beschreibung:
% Diese Funktion führt die Generierung und Spiecherung der Trainigsdaten
% durch. Dabei wird die bereits hergeleitete Zustandsraumdarstellung des
% 2-FHG Roboters verwendet. Die Speicherung erfolgt im Folgenden
% Verzeichnis: Studienarbeit_Deep_Lagrangian_Networks\Training_Data\MATLAB_Simulation
% -------------------------------------------------------------

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

% Diese Funktion erstellt eine Matrix mit zufälligen Eingangssignalen. Die
% dazu verwendeten Intervalle der Zufallsparameter können innerhalb der
% Funktion verändert werden.
function [u_mat] = create_random_input(t_u)
    % Intervalle Zufallsparameter (können verändert werden)
    gain = [0.1, 2];
    frequency = [0.1, 2];
    offset = [t_u(1) + 1, t_u(end) - 1];
    
    % Eingangsgrößen Kraft F
    u_mat(1, :) = t_u.*0;  % kein Eingang
    u_mat(2, :) = random_sign()*random_param(gain(1), gain(2))*heaviside(t_u - random_param(offset(1), offset(2)));    % Zufällige Sprungfunktion
    u_mat(3, :) = random_sign()*random_param(gain(1), gain(2))*sin(random_param(frequency(1), frequency(2))*(t_u - random_param(0, pi)));   % Zufällige Sinus Funktion
    u_mat(4, :) = random_sign()*random_param(gain(1), gain(2))*square(random_param(frequency(1), frequency(2))*t_u);   % Rechteckfunktion
    u_mat(5, :) = random_sign()*random_param(gain(1), gain(2))*sawtooth(random_param(frequency(1), frequency(2))*t_u);  % Sägzahnfunktion

end

% Diese Funktion erstellt zufällige Anfangswerte in bestimmte Intervallen.
% Die Intervalle können manuell in der Funktion angepasst werden.
function x_0 = random_initial_values()
    % Anfangswerte
    r_0 = random_param(-1, 1);  % Sollte immer größer als l sein
    phi_0 = random_param(-2*pi, 2*pi);
    r_p_0 = random_param(-0.5, 0.5);
    phi_p_0 = random_param(-0.5, 0.5);

    x_0 = [r_0; phi_0; r_p_0; phi_p_0]; % Vektor der Anfangswerte
end

clc
clear
close all

%% Definition der Systemparameter

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

% Anfangswerte und Simulationszeit
t_span = [0 10];    % Simulationszeit

% sollen Plots angezeigt werden?
showplots = true;

% sollen Simulationsdaten gespeichert werden
savedata = true;

%% Eingangssignale

% Zeitsignal
t_u = linspace(t_span(1), t_span(2), 1000);

% Eingangssignale Verläufe
uF_vec = create_random_input(t_u);
utau_vec = create_random_input(t_u);

%% Lösung der ODE mit variablen Stellgrößenverläufen und Eingangssignalen

% Anzahl der Stellgrößenverläufe in u
n = size(uF_vec);
n = n(1);

% Strukturarray zur Speicherung von Daten
simData(n^2) = struct();
counter = 1;

% Alle Stellgrößenverläufe durchgehen
for i = 1:n
    for j = 1:n
        % Anfangswerte
        x_0 = random_initial_values();

        % Stellgrößen
        F_vec = [t_u; uF_vec(i, :)];
        tau_vec = [t_u; utau_vec(j, :)];

        % ODE-Funktion mit Parametern
        odefun = @(t, x) ODE_2_FHG_Robot(t, x, F_vec, tau_vec, l_m, m_kg, mL_kg, J_kgm2);
        
        % Solver zur Lösung der DGL
        options = odeset('MaxStep', 0.01, 'Stats', 'on');
        [t, x] = ode45(odefun, t_span, x_0, options);

        % Speichern
        simData(counter).t = t;
        simData(counter).x = x;
        simData(counter).tau = utau_vec(j, :);
        simData(counter).F = uF_vec(i, :);

        counter = counter + 1;

    end
end

%% Beschleunigungsterme berechnen (numerisch)

% Schleife, die alle Simulationen durchgeht
for i = 1:n^2
    % Gradienten berechnen
    simData(i).r_pp = gradient(simData(i).x(:, 3), simData(i).t);
    simData(i).phi_pp = gradient(simData(i).x(:, 4), simData(i).t);
end

%% Simulationsergebnisse Speichern (Daten in Trainings- und Testdaten aufteilen)

% Zufälligen Testdatensatz herausziehen
test_index = randi([1, n^2]);   % Zufälliger Index für Testdaten
features_test = [simData(test_index).x(:, 1), simData(test_index).x(:, 2), simData(test_index).x(:, 3), simData(test_index).x(:, 4), interp1(t_u, simData(test_index).F, simData(test_index).t), interp1(t_u, simData(test_index).tau, simData(test_index).t)];
labels_test = [simData(test_index).r_pp, simData(test_index).phi_pp];

% Trainingsdaten in Features und Labels aufteilen.
features_training = [simData(1).x(:, 1), simData(1).x(:, 2), simData(1).x(:, 3), simData(1).x(:, 4), interp1(t_u, simData(1).F, simData(1).t), interp1(t_u, simData(1).tau, simData(1).t)];
labels_training = [simData(1).r_pp, simData(1).phi_pp];
for i = 2:n^2
    if i ~= test_index
        features_training = [features_training;
            simData(i).x(:, 1), simData(i).x(:, 2), simData(i).x(:, 3), simData(i).x(:, 4), interp1(t_u, simData(i).F, simData(i).t), interp1(t_u, simData(i).tau, simData(i).t)];
        labels_training = [labels_training;
            simData(i).r_pp, simData(i).phi_pp];
    end
end

if savedata == true
    % Pfad dieses Skripts
    my_path = fileparts(mfilename('fullpath'));

    % Zielordner relativ zum Skriptpfad
    target_folder = fullfile(my_path, '..', 'Training_Data', 'MATLAB_Simulation');
    target_folder = fullfile(target_folder); % Pfad normalisieren

    % Datei speichern
    time_stamp = string(datetime('now', 'Format', 'yyyy_MM_dd_HH_mm_ss'));
    dateiName = 'SimData__' + time_stamp + '.mat';
    full_path = fullfile(target_folder, dateiName);
    save(full_path, 'features_training', 'labels_training', 'features_test', 'labels_test');
end

%% Plots erstellen

% Plots 
if showplots == true
    for i = 1:n^2
        % Plot erstellen
        figure(i);
    
        % Oberer linker Plot
        subplot(2,2,1); % 2 Zeilen, 1 Spalte, oberer Plot
        plot(simData(i).t, simData(i).x(:, 1), 'b', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Weg [m]');
        grid on;
        title('Position r(t)');
        
        % Oberer rechter Plot
        subplot(2,2,2); % 2 Zeilen, 1 Spalte, unterer Plot
        plot(simData(i).t, simData(i).x(:, 2), 'b', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Winkel [rad]');
        grid on;
        title('Winkel phi(t)');

        % Unterer linker Plot
        subplot(2,2,3); % 2 Zeilen, 1 Spalte, unterer Plot
        plot(t_u, simData(i).F, 'r', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Kraft [N]');
        grid on;
        title('Antriebskraft F');

        % Unterer rechter Plot
        subplot(2,2,4); % 2 Zeilen, 1 Spalte, unterer Plot
        plot(t_u, simData(i).tau, 'r', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Moment [Nm]');
        grid on;
        title('Antriebsdrehmoment tau');
    
    end
end
