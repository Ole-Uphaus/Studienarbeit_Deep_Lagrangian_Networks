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

%% Kinematik

% Erster Arm

