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

% Funktion, die das trainierte Modell als ODE definiert
function [dxdt] = ODE_Neral_Network(t, x, net, F_vec, tau_vec, scaler)
    dxdt = zeros(4, 1); % 4 Zustände

    % aktuelle Stellgrößen auslesen (Der Zeitvektor ist jeweils in F_vec
    % und tau_vec enthalten -> f_vec(1, :) - Zeitvektor)
    F = interp1(F_vec(1, :), F_vec(2, :), t);
    tau = interp1(tau_vec(1, :), tau_vec(2, :), t);

    % neuronales Netz auswerten
    input_data = [x(1);
        x(2);
        x(3);
        x(4);
        F;
        tau];

    input_data_scaled = (input_data - scaler.mean_f') ./ scaler.scale_f';   % Eingangsdaten skalieren (wie bei Training des NN)

    dlInput = dlarray(input_data_scaled, 'CB');
    dlOutput = predict(net, dlInput);
    predictions = extractdata(dlOutput);

    predictions_scaled = predictions .* scaler.scale_l' + scaler.mean_l';

    % Ableitungen des Zustandsvektors übergeben
    dxdt(1) = x(3);
    dxdt(2) = x(4);
    dxdt(3) = predictions_scaled(1);
    dxdt(4) = predictions_scaled(2);

end

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
options = odeset('MaxStep', 0.1, 'Stats', 'on');
[t_zrd, x_zrd] = ode45(odefun_1, t_span, x_0, options);

%% DGL lösen (mit trainiertem Modell aus Python)

% neuronales Netz importieren
network_name = "20250327_142624_feedforward_model.onnx";
network_path = fullfile(my_path, '..', 'Training_Models', 'Feedforward_NN', 'Saved_Models', network_name);

net = importNetworkFromONNX(network_path, 'InputDataFormats', {'BC'});

% Scaler importieren
scaler_name = "20250327_142624_scaler.mat";
scaler_path = fullfile(my_path, '..', 'Training_Models', 'Feedforward_NN', 'Saved_Models', scaler_name);

scaler = load(scaler_path);

% ODE-Funktion mit Parametern
odefun_2 = @(t, x) ODE_Neral_Network(t, x, net, F_vec, tau_vec, scaler);

% Solver zur Lösung der DGL
options = odeset('MaxStep', 0.1, 'Stats', 'on');
[t_NN, x_NN] = ode45(odefun_2, t_span, x_0, options);

%% Plotten

figure();

% Oberer Plot (r(t))
subplot(2,1,1); % 2 Zeilen, 1 Spalte, oberer Plot
plot(t_zrd, x_zrd(:, 1), 'b', 'LineWidth', 1.5, 'DisplayName', 'Zustandsraummodell');
hold on;
plot(t_NN, x_NN(:, 1), 'r', 'LineWidth', 1.5, 'DisplayName', 'Neuronales Netzwerk');
xlabel('Zeit [s]');
ylabel('Weg [m]');
grid on;
hold off;
legend show
title('Position r(t)');

% Unterer Plot (phi(t))
subplot(2,1,2); % 2 Zeilen, 1 Spalte, unterer Plot
plot(t_zrd, x_zrd(:, 2), 'b', 'LineWidth', 1.5, 'DisplayName', 'Zustandsraummodell');
hold on;
plot(t_NN, x_NN(:, 2), 'r', 'LineWidth', 1.5, 'DisplayName', 'Neuronales Netzwerk');
xlabel('Zeit [s]');
ylabel('Winkel [rad]');
grid on;
hold off;
legend show
title('Winkel phi(t)');