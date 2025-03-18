% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      18.03.2025
% Beschreibung:
% Dieses Skript soll die differentialgleichungen des 2-FHG Roboters
% mithilfe der Lagrangeschen Gleichungen 2.Art herleiten. Außerdem soll
% eine Zustandsraumdarstellung hergeleitet werden, die säter zur Simulation
% verwendet werden kann. Damit nicht in jedem Simulationsschritt die
% Lagrange Gleichungen gelöst werden müssen, wird die Lösung in diesem
% Skript durchgeführt.
% -------------------------------------------------------------

% Symbolische variablen definieren
syms r phi r_p phi_p r_pp phi_pp F tau
syms J m mL l

% Lagrange Funktion aufstellen
U = 0;
T = 1/2*J*phi_p^2 + 1/2*(m + mL)*r_p^2 + 1/2*(m*(r - l)^2)*phi_p^2 + 1/2*(mL*r^2)*phi_p^2;
L = T - U;

% Lagrange Gleichungen 2.Art
q = [r; phi];
q_p = [r_p; phi_p];
jac_1 = jacobian(L, q_p).';
jac_2 = jacobian(L, q).';

Lagr_eq = simplify(diff(jac_1, r)*r_p + diff(jac_1, r_p)*r_pp + diff(jac_1, phi)*phi_p + diff(jac_1, phi_p)*phi_pp - jac_2);

% Zustandsraumdarstellung
x_p = [q_p; solve(Lagr_eq(1) == F, r_pp); solve(Lagr_eq(2) == tau, phi_pp)]
