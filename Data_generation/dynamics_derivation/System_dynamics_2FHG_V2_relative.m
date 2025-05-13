% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      30.04.2025
% Beschreibung:
% Dieses Skript soll die differentialgleichungen des 2-FHG Roboters
% mithilfe der Lagrangeschen Gleichungen 2.Art herleiten. In diesem Skript
% wird im Gegensatz zum ersten Skript die Dynamik eines weiteren 2 FHG
% Roboters hergeleitet (Siehe Aufzeichnungen).
% -------------------------------------------------------------
clc
clear
close all

% Symbolische Variablen definieren
syms phi_1 phi_2 phi_1_p phi_2_p phi_1_pp phi_2_pp real
syms m_1 m_2 J_1 J_2 l_1 l_2 g real

%% Kinematik (Translation)

% Erster Arm
r_s1_vec = [l_1/2*sin(phi_1);
    -l_1/2*cos(phi_1)];

v_s1_vec = diff(r_s1_vec, phi_1)*phi_1_p;
v_s1_squared = simplify(expand(v_s1_vec'*v_s1_vec));

% Zweiter Arm
r_s2_vec = [l_1*sin(phi_1) + l_2/2*sin(phi_1 + phi_2);
    -l_1*cos(phi_1) - l_2/2*cos(phi_1 + phi_2)];

v_s2_vec = diff(r_s2_vec, phi_1)*phi_1_p + diff(r_s2_vec, phi_2)*phi_2_p;
v_s2_squared = simplify(expand(v_s2_vec'*v_s2_vec));

%% Kinematik (Rotation)

% Erster Arm
theta_1_p = phi_1_p;

% Zweiter Arm
theta_2_p = phi_1_p + phi_2_p;

%% Lagrange Funktion

% Kinetische Energie
T = 1/2*m_1*v_s1_squared + 1/2*m_2*v_s2_squared + 1/2*J_1*theta_1_p^2 + 1/2*J_2*theta_2_p^2;

% Potentielle Energie
U = m_1*g*r_s1_vec(2, 1) + m_2*g*r_s2_vec(2, 1);

% Lagrange Funktion
L = T - U;

%% Lagrange Gleichungen 2. Art

% Minimalkoordinaten
q = [phi_1, phi_2]';
q_p = [phi_1_p, phi_2_p]';
q_pp = [phi_1_pp, phi_2_pp]';

% Lagrange Gleichung
jac_1 = jacobian(L, q_p).';
jac_2 = jacobian(L, q).';

Lagr_eq = simplify(jacobian(jac_1, q)*q_p + jacobian(jac_1, q_p)*q_pp - jac_2);

%% Massen-, Coriolis- und Gravitationsterme bestimmen

% Massenmatrix
M = jacobian(Lagr_eq, q_pp)

% Coriolis- und Gravitationsterme
cor_and_grav = simplify(Lagr_eq - M*q_pp);

% Gravitationsterme
grav = simplify(subs(cor_and_grav, [phi_1_p, phi_2_p], [0, 0]))

% Coriolisterme
cor = simplify(cor_and_grav - grav)