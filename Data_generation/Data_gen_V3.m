% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      17.04.2025
% Beschreibung:
% Dies ist eine komplett neue Version des Skriptes zur Generierung der
% Trainingsdaten. Ich werde in diesem skript nicht so viel dem Zufall
% überlassen und Solltrajektorien für r und phi vorgeben (mit einem
% gewissen Zufallsanteil). Aus diesen Solltrajektorien werde ich
% anschließend die Ableitungen berechnen und die inverse Dynamik auswerten.
% Zur Generierung der TRainingsdaten muss somit keine ODE gelöst werden.
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
function x_0 = random_init(n, lower_barrier, upper_barrier)
    % leeren Vektor
    x_0 = zeros(n, 1);
    for i = 1:n
        x_0(i) = random_param(lower_barrier, upper_barrier);
    end
end

%% Parameterdefinition

% Anzahl der Zeitschritte pro Bewegung
smples_per_run = 100;

% Anzahl der voneinander unabhängigen Bewegungen
number_runs = 20;

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

%% Zufällige Anfangsbedingungen 

% Lageebene
r_0 = random_init(number_runs, 0, 0.5); % Intervall [0, 0.5]
phi_0 = random_init(number_runs, 0, 2*pi);  % Intervall [0, 2*pi]







% % Anzahl der Gelenke
% n_dof = 1;
% 
% % Zeitpunkte der Wegpunkte
% waypointTimes = [0 5];
% 
% % Abtastzeiten
% t_vec = linspace(0, 5, 100);
% 
% % Wegpunkte
% waypoints = [0 3];
% 
% % Berechnung der Trajektorie
% [q, qd, qdd] = quinticpolytraj(waypoints, waypointTimes, t_vec);
% 
% % Plotten
% figure();
% hold on
% plot(t_vec, q, 'b', 'LineWidth', 1.5, DisplayName="q")
% plot(t_vec, qd, 'r', 'LineWidth', 1.5, DisplayName="qd")
% plot(t_vec, qdd, 'g', 'LineWidth', 1.5, DisplayName="qdd")
% xlabel('Zeit [s]');
% ylabel('Position');
% grid on;
% title('Position');
