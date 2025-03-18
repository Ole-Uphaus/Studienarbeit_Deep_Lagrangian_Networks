% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      18.03.2025
% Beschreibung:
% Diese Funktion beinhaltet die Zustandraumdarstellung des 2-FHG Roboters
% aus der Vorlesung. Sie wird später zur Generierung der Trainigsdaten
% verwendet. Die Stellgrößenverläufe müssen zuvor in der Hauptdatei als
% u-t-Verlauf definiert werden. Die Reitschrittweite sollte dabei mit der
% Simulationsschrittweite übereinstimmen.
% -------------------------------------------------------------

function [dxdt] = ODE_2_FHG_Robot(t, x, F_vec, tau_vec, l, m, mL, J)
    dxdt = zeros(4, 1); % 4 Zustände

    % aktuelle Stellgrößen auslesen (Der Zeitvektor ist jeweils in F_vec
    % und tau_vec enthalten -> f_vec(1, :) - Zeitvektor)
    F = interp1(F_vec(1, :), F_vec(2, :), t);
    tau = interp1(tau_vec(1, :), tau_vec(2, :), t);

    % Zustandsraumdarstellung
    r = x(1);
    phi = x(2);
    r_p = x(3);
    phi_p = x(4);

    dxdt(1) = r_p;
    dxdt(2) = phi_p;
    dxdt(3) = (F - l*m*phi_p^2 + m*phi_p^2*r + mL*phi_p^2*r)/(m + mL);
    dxdt(4) = (tau + r_p*(m*phi_p*(2*l - 2*r) - 2*mL*phi_p*r))/(J + mL*r^2 + m*(l - r)^2);
end

