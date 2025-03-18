% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      18.03.2025
% Beschreibung:
% Diese Funktion führt die Generierung und Spiecherung der Trainigsdaten
% durch. Dabei wird die bereits hergeleitete Zustandsraumdarstellung des
% 2-FHG Roboters verwendet. 
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
    frequency = [0.1, 5];
    offset = [t_u(1) + 1, t_u(end) - 1];
    
    % Eingangsgrößen Kraft F
    u_mat(1, :) = t_u.*0;  % kein Eingang
    u_mat(2, :) = random_sign()*random_param(gain(1), gain(2))*heaviside(t_u - random_param(offset(1), offset(2)));    % Zufällige Sprungfunktion
    u_mat(3, :) = random_sign()*random_param(gain(1), gain(2))*sin(random_param(frequency(1), frequency(2))*(t_u - random_param(0, pi)));   % Zufällige Sinus Funktion
    u_mat(4, :) = random_sign()*random_param(gain(1), gain(2))*square(random_param(frequency(1), frequency(2))*t_u);   % Rechteckfunktion
    u_mat(5, :) = random_sign()*random_param(gain(1), gain(2))*sawtooth(random_param(frequency(1), frequency(2))*t_u);  % Sägzahnfunktion

end

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

        counter = counter + 1;

    end
end

%% Plots erstellen

% sollen Plots angezeigt werden?
showplots = true;

if showplots == true
    for i = 1:n^2
        % Plot erstellen
        figure(i);
    
        % Oberer Plot
        subplot(2,1,1); % 2 Zeilen, 1 Spalte, oberer Plot
        plot(simData(i).t, simData(i).x(:, 1), 'b', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Weg [m]');
        title('Position r(t)');
        
        % Unterer Plot
        subplot(2,1,2); % 2 Zeilen, 1 Spalte, unterer Plot
        plot(simData(i).t, simData(i).x(:, 2), 'r', 'LineWidth', 1.5);
        xlabel('Zeit [s]');
        ylabel('Winkel [rad]');
        title('Winkel phi(t)');
    
    end
end
