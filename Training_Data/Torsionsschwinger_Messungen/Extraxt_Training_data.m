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

% Schwellenwert für Geschwindigkeit
threshold_v = 2.e-1;

% Daten abspeichern
savedata = true;

%% Signale filtern

% Winkel in Radiant umrechnen
phi1_rad = phi1;
phi2_rad = phi2;

% Filter vorbereiten
dt = mean(diff(t_ges));
fs = 1 / dt; % Abtastrate
[b, a] = butter(2, fc / (fs/2));

% Filtern der Positionssignale und daraus dann die Ableitungen bestimmen
phi1_filt = filtfilt(b, a, phi1_rad);
phi2_filt = filtfilt(b, a, phi2_rad);

%% Geschwindigkeiten und Beschleunigungen berechnen

% Geschwindigkeiten
phi1p_filt = gradient(phi1_filt, dt);
phi2p_filt = gradient(phi2_filt, dt);

% Beschleunigungen
phi1pp_filt = gradient(phi1p_filt, dt);
phi2pp_filt = gradient(phi2p_filt, dt);

% Test: Integral der Beschleunigungen berechnen
phi1_p_int = cumtrapz(phi1pp_filt)*dt;

%% Ursprungssignale Plotten

% Positionen
figure()
subplot(2,1,1);
plot(t_ges, phi1_filt, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1');
grid on;
title('phi1(t)');

subplot(2,1,2);
plot(t_ges, phi2_filt, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2');
grid on;
title('phi2(t)');

% Geschwindigkeiten
figure()
subplot(2,1,1);
plot(t_ges, phi1p_filt, 'LineWidth', 1.5);
hold on
plot(t_ges, phi1_p_int, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1p');
grid on;
title('phi1p(t)');

subplot(2,1,2);
plot(t_ges, phi2p_filt, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2p');
grid on;
title('phi2p(t)');

% Beschleunigungen
figure()
subplot(2,1,1);
plot(t_ges, phi1pp_filt, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1pp');
grid on;
title('phi1pp(t)');

subplot(2,1,2);
plot(t_ges, phi2pp_filt, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2pp');
grid on;
title('phi2pp(t)');

% Drehmoment
figure()
plot(t_ges, u, 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('tau');
grid on;
title('tau(t)');


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

%% Zusammenfassen und Trainings und Testdaten erstellen

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

%% Datenpunkte im Stillstand löschen

% Trainingsdaten (Geschwindigkeit == 0 -> Datenpunkt löschen)
training_delete_idx = find(abs(features_training(:, 3)) < threshold_v | abs(features_training(:, 4)) < threshold_v);

features_training(training_delete_idx, :) = [];
labels_training(training_delete_idx, :) = [];

% Testdaten (Geschwindigkeit == 0 -> Datenpunkt löschen)
test_delete_idx = find(abs(features_test(:, 3)) < threshold_v | abs(features_test(:, 4)) < threshold_v);

features_test(test_delete_idx, :) = [];
labels_test(test_delete_idx, :) = [];

% Samples vektoren erstellen
samples_training = size(features_test, 1):(size(features_test, 1) + size(features_training, 1) - 1);
samples_test = 0:(size(features_test, 1) - 1);

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
plot(samples_test, features_test(:, 1), 'LineWidth', 1.5);
plot(samples_training, features_training(:, 1), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1');
grid on;
title('phi1(t)');

subplot(2,1,2);
% plot(t_ges, phi2_rad, 'LineWidth', 1.5);
hold on
plot(samples_test, features_test(:, 2), 'LineWidth', 1.5);
plot(samples_training, features_training(:, 2), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2');
grid on;
title('phi2(t)');

% Geschwindigkeiten
figure()
subplot(2,1,1);
plot(samples_test, features_test(:, 3), 'LineWidth', 1.5);
hold on
plot(samples_training, features_training(:, 3), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1p');
grid on;
title('phi1p(t)');

subplot(2,1,2);
plot(samples_test, features_test(:, 4), 'LineWidth', 1.5);
hold on
plot(samples_training, features_training(:, 4), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2p');
grid on;
title('phi2p(t)');

% Beschleunigungen
figure()
subplot(2,1,1);
plot(samples_test, labels_test(:, 1), 'LineWidth', 1.5);
hold on
plot(samples_training, labels_training(:, 1), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi1pp');
grid on;
title('phi1pp(t)');

subplot(2,1,2);
plot(samples_test, labels_test(:, 2), 'LineWidth', 1.5);
hold on
plot(samples_training, labels_training(:, 2), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('phi2pp');
grid on;
title('phi2pp(t)');

% Drehmoment
figure()
plot(samples_test, features_test(:, 5), 'LineWidth', 1.5);
hold on
plot(samples_training, features_training(:, 5), 'LineWidth', 1.5);
xlabel('Zeit [s]');
ylabel('tau');
grid on;
title('tau(t)');
