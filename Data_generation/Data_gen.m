% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      18.03.2025
% Beschreibung:
% Diese Funktion führt die Generierung und Spiecherung der Trainigsdaten
% durch. Dabei wird die bereits hergeleitete Zustandsraumdarstellung des
% 2-FHG Roboters verwendet. 
% -------------------------------------------------------------
clc
clear

%% Definition der Systemparameter und Initialisierung

% Systemparameter
m_kg = 5;   % Masse des Arms
mL_kg = 2;  % Masse der Last
J_kgm2 = 0.4;  % gesamte Rotationsträgheit
l_m = 0.25; % Schwerpunktsabstand (Arm - Last)

% Anfangswerte und Simulationszeit
t_span = [0 10];    % Simulationszeit

r_0 = 0.5;  % Sollte immer größer als l sein
phi_0 = 0;
r_p_0 = 0;
phi_p_0 = 0;
x_0 = [r_0; phi_0; r_p_0; phi_p_0]; % Vektor der Anfangswerte

%% Eingangssignale

% Zeitsignal
t_u = linspace(t_span(1), t_span(2), 1000);

u_zero = t_u.*0;    % kein Eingang
u_step = heaviside(t_u - 2);    % Sprungfunktion
u_sin = sin(t_u);   % Sinus Funktion
u_rectangle = square(2*pi*0.5*t_u);   % Rechteckfunktion
u_sawtooth = sawtooth(2*pi*0.5*t_u);  % Sägzahnfunktion

% Stellgrößen
F_vec = [t_u; u_sawtooth];
tau_vec = [t_u; u_step];

%% Differentialgleichung lösen

% ODE-Funktion mit Parametern
odefun = @(t, x) ODE_2_FHG_Robot(t, x, F_vec, tau_vec, l_m, m_kg, mL_kg, J_kgm2);

% Solver zur Lösung der DGL
options = odeset('MaxStep', 0.01, 'Stats', 'on');
[t, x] = ode45(odefun, t_span, x_0, options);

%% Plots erstellen

% Plot 1. Signal in Figure 1
figure(1);                   % Neues Fenster (Figure 1)
plot(t, x(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('Weg [m]');
title('Position r(t)');
grid on;

% Plot 2. Signal in Figure 2
figure(2);                   % Neues Fenster (Figure 2)
plot(t, x(:, 2), 'r', 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('Winkel [rad]');
title('Winkel phi(t)');
grid on;