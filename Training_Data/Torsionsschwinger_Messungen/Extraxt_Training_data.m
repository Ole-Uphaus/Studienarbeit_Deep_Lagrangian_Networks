% -------------------------------------------------------------
% Autor:      Ole Uphaus
% Datum:      20.06.2025
% Beschreibung:
% Dieses Skript nutz die Messungen von Sven, um geeignete Trainingsdaten
% für das Delan Modell zu erstellen. Mit diesen kann dann später trainiert
% werden.
% -------------------------------------------------------------

clc
clear
close all

%% Parameterderfinition

% Messdaten laden
load("Messung_Torsionsschwinger.mat")

% Grenzfrequenzen Filter
fc = 1;

% Resampling rate
r_resample = 10;

% Index, bis zu dem Testdaten gesammelt werden
test_idx = 2100;

% Daten abspeichern
savedata = false;

%% Signale filtern

% Winkel in Radiant umrechnen
phi1_rad = phi1 / 180 * pi;
phi2_rad = phi2 / 180 * pi;

% Filter vorbereiten
fs = 1 / mean(diff(t_ges)); % Abtastrate
[b, a] = butter(2, fc / (fs/2));

% Filtern der Positionssignale und daraus dann die Ableitungen bestimmen
phi1_filt = filtfilt(b, a, phi1_rad);
phi2_filt = filtfilt(b, a, phi2_rad);

%% Geschwindigkeiten und Beschleunigungen berechnen

% Geschwindigkeiten
phi1p_filt = gradient(phi1_filt) ./ gradient(t_ges);
phi2p_filt = gradient(phi2_filt) ./ gradient(t_ges);

% Beschleunigungen
phi1pp_filt = gradient(phi1p_filt) ./ gradient(t_ges);
phi2pp_filt = gradient(phi2p_filt) ./ gradient(t_ges);

%% Signale downsamplen

% Resamplen
phi1_filt_res = downsample(phi1_filt, r_resample);
phi2_filt_res = downsample(phi2_filt, r_resample);

phi1p_filt_res = downsample(phi1p_filt, r_resample);
phi2p_filt_res = downsample(phi2p_filt, r_resample);

phi1pp_filt_res = downsample(phi1pp_filt, r_resample);
phi2pp_filt_res = downsample(phi2pp_filt, r_resample);

t_ges_res = downsample(t_ges, r_resample);

u_res = downsample(u, r_resample);

%% Zusammenfassen unt Trainings und Testdaten erstellen

% Zusammenfassen
features_ges = [phi1_filt_res, phi2_filt_res, phi1p_filt_res, phi2p_filt_res, u_res, zeros(size(u_res))];
labels_ges = [phi1pp_filt_res, phi2pp_filt_res];

% Aufteilen in Trainings und Testdaten
features_training = features_ges((test_idx + 1):end, :);
labels_training = labels_ges((test_idx + 1):end, :);
t_vec_training = t_ges_res((test_idx + 1):end, :);

features_test = features_ges(1:test_idx, :);
labels_test = labels_ges(1:test_idx, :);
t_vec_test = t_ges_res(1:test_idx, :);

% Massen und Coriolisterme erstellen (hier unbekannt - gleich null setzen
% für Plot später)
Mass_Cor_test = zeros(size(features_test, 1), 9);

%% Abspeichern

if savedata == true
    % Pfad dieses Skripts
    my_path = fileparts(mfilename('fullpath'));

    % Datei speichern
    dateiName = 'Measuring_data_Training_Torsionsschwinger.mat';   
    full_path = fullfile(my_path, dateiName);
    save(full_path, 'features_training', 'labels_training', 'features_test', 'labels_test', "Mass_Cor_test");
end

%% Plotten

% Positionen
figure()
subplot(2,1,1);
% plot(t_ges, phi1_rad, 'LineWidth', 1.5);
hold on
plot(t_vec_test, features_test(:, 1), 'LineWidth', 1.5);
plot(t_vec_training, features_training(:, 1), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1');
grid on;
title('phi1(t)');

subplot(2,1,2);
% plot(t_ges, phi2_rad, 'LineWidth', 1.5);
hold on
plot(t_vec_test, features_test(:, 2), 'LineWidth', 1.5);
plot(t_vec_training, features_training(:, 2), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2');
grid on;
title('phi2(t)');

% Geschwindigkeiten
figure()
subplot(2,1,1);
plot(t_vec_test, features_test(:, 3), 'LineWidth', 1.5);
hold on
plot(t_vec_training, features_training(:, 3), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1p');
grid on;
title('phi1p(t)');

subplot(2,1,2);
plot(t_vec_test, features_test(:, 4), 'LineWidth', 1.5);
hold on
plot(t_vec_training, features_training(:, 4), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2p');
grid on;
title('phi2p(t)');

% Beschleunigungen
figure()
subplot(2,1,1);
plot(t_vec_test, labels_test(:, 1), 'LineWidth', 1.5);
hold on
plot(t_vec_training, labels_training(:, 1), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1pp');
grid on;
title('phi1pp(t)');

subplot(2,1,2);
plot(t_vec_test, labels_test(:, 2), 'LineWidth', 1.5);
hold on
plot(t_vec_training, labels_training(:, 2), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2pp');
grid on;
title('phi2pp(t)');

% Drehmoment
figure()
plot(t_vec_test, features_test(:, 5), 'LineWidth', 1.5);
hold on
plot(t_vec_training, features_training(:, 5), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('tau');
grid on;
title('tau(t)');
