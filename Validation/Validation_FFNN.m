% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      21.03.2025
% Beschreibung:
% Dieses Skript soll das Mithilfe von pytorch tranierte Modell validieren.
% Dazu wird es genutzt. Um eine Differentialgleiochung zu lösen.
% Anschließend werden die realen Ergebnisse, mit denen des Trainierten
% Modells verglichen.
% -------------------------------------------------------------

clear
clc
close all

%% Definition der Systemparameter

% Dateipfad von Funktion hinzufügen
my_path = fileparts(mfilename('fullpath'));
function_path = fullfile(my_path, '..', 'Data_generation');

addpath(function_path)

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

% Anfangswerte und Simulationszeit
t_span = [0 10];    % Simulationszeit

% Anfangswerte
r_0 = 0.5;
phi_0 = 0;
r_p_0 = 0;
phi_p_0 = 0;

x_0 = [r_0; phi_0; r_p_0; phi_p_0]; % Vektor der Anfangswerte

%% Eingangssignale

% Zeitsignal
t_u = linspace(t_span(1), t_span(2), 1000);

% Eingangssignale Verläufe
uF_vec = 0.*t_u;
utau_vec = heaviside(t_u - 3);

% Stellgrößen
F_vec = [t_u; uF_vec];
tau_vec = [t_u; utau_vec];

%% DGL lösen (mit analytischer Zustandsraumdarstellung)

% ODE-Funktion mit Parametern
odefun_1 = @(t, x) ODE_2_FHG_Robot(t, x, F_vec, tau_vec, l_m, m_kg, mL_kg, J_kgm2);

% Solver zur Lösung der DGL
options = odeset('MaxStep', 0.01, 'Stats', 'on');
[t_zrd, x_zrd] = ode45(odefun_1, t_span, x_0, options);

%% DGL lösen (mit trainiertem Modell aus Python)

% neuronales Netz importieren
network_name = "20250321_111518_feedforward_model.onnx";
network_path = fullfile(my_path, '..', 'Training_Models', 'Feedforward_NN', 'Saved_Models', network_name);

net = importNetworkFromONNX(network_path, 'InputDataFormats', {'BC'});

%% Plotten

figure();

% Oberer Plot (r(t))
subplot(2,1,1); % 2 Zeilen, 1 Spalte, oberer Plot
plot(t_zrd, x_zrd(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('Weg [m]');
grid on;
title('Position r(t)');

% Unterer Plot (phi(t))
subplot(2,1,2); % 2 Zeilen, 1 Spalte, unterer Plot
plot(t_zrd, x_zrd(:, 2), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('Winkel [rad]');
grid on;
title('Winkel phi(t)');