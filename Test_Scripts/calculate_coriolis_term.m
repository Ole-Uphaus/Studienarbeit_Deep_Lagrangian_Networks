% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      09.05.2025
% Beschreibung:
% In diesem Skript werde ich einmal die Umformung aus dem DeLaN Paper
% durchführen. Dazu werde ich mithilfe von symbolischen Variablen, die
% Richtigkeit der Umformung prüfen, um sicherzugehen, dass auchich in
% meinen Berechnungen die richtigen Operationen durchführe.
% -------------------------------------------------------------

clc
clear
close all

% Es geht um folgenden Ausdruck: d/dq(qpT*H*qp) = qpT*dH/dq*qp

% Symbolische Variablen definieren
syms q1 q2 q1_p q2_p real

% Vektorausdrücke
q = [q1;
    q2];    % Spaltenvektor
q_p = [q1_p;
    q2_p];  % Spaltenvektor

% Matrix H(q) definieren
H = [2*q1, q2^2;
    q2^2, q1*q2]

%% 1. Moglichkeit: Ausmultiplizieren und am Ende ableiten

% Inneres Produkt bilden (qpT*H*qp)
qpT_H_qp = q_p'*H*q_p;   % Skalerer Ausdruck

% Ableitung nach q bilden
result_1 = jacobian(qpT_H_qp, q)    % Zeilenvektor

%% 2. Möglichkeit: Zuerst dh/dq berechnen und dann ausmultiplizieren

% dH/dq berechnen
dH_dq1 = diff(H, q1);    % Ableitungen einzeln berechnen , da jacobian keine Matrizen als eingang erlaubt
dH_dq2 = diff(H, q2);

dH_dq = cat(3, dH_dq1, dH_dq2);  % Einzelne Ableitung zu 3D Matrix (Würfel zusammensetzen)

% Ausmultiplizieren qpT*dH/dq*qp
dHdq1_qp = q_p'*dH_dq(:, :, 1)*q_p;
dHdq2_qp = q_p'*dH_dq(:, :, 2)*q_p;

result_2 = [dHdq1_qp, dHdq2_qp]

%% Möglichkeit 3: Zuerst dh/dq berechnen und dann ausmultiplizieren (so, wie ich es bis jetzt implementiert habe)

% Ausmultiplizieren dH/dq*qp (in dq Richtung - letzte richtung)
dH1dq_qp = reshape(dH_dq(1, :, :), 2, 2)*q_p;
dH2dq_qp = reshape(dH_dq(2, :, :), 2, 2)*q_p;

dHdq_qp = [dH1dq_qp'; dH2dq_qp'];

% Ausmultiplizieren qpT*dH/dq*qp
result_3 = q_p'*dHdq_qp

%% Übereinstimmung prüfen

% Ergebnisse 1 und 2
if isequal(result_1, result_2)
    disp('Ergebnisse 1 und 2 stimmen überein.')
else
    disp('Ergebnisse 1 und 2 stimmen nicht überein.')
end

% Ergebnisse 1 und 3
if isequal(result_1, result_3)
    disp('Ergebnisse 1 und 3 stimmen überein.')
else
    disp('Ergebnisse 1 und 3 stimmen nicht überein.')
end