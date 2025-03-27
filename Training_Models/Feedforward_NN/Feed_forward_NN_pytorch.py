'''
Autor:      Ole Uphaus
Datum:      20.03.2025
Beschreibung:
Dieses Skript soll auf Grundlage der Simuleirten Trainingsdaten ein neuronales Netz trainieren. Dies ist jedoch nur ein Feed-Forward Netz, das keine Informationen über lagrange Gleichungen enthält. Die benötigten Trainingsdaten werden aus einem .mat File extrahiert.
'''

import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import onnx

def extract_training_data(file_name):
    # Pfad des aktuellen Skriptes
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zum Datenordner von hier aus
    # Wir müssen zwei Ebenen hoch und dann in den Zielordner
    data_path = os.path.join(script_path, '..', '..', 'Training_Data', 'MATLAB_Simulation', file_name)

    # Pfad normieren
    data_path = os.path.normpath(data_path)

    # Daten extrahieren
    data = scipy.io.loadmat(data_path)

    features_training = data['features_training']
    labels_training = data['labels_training']
    features_test = data['features_test']
    labels_test = data['labels_test']

    return features_training, labels_training, features_test, labels_test

# Erstellung neuronales Netz (Klasse)
class Feed_forward_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feed_forward_NN, self).__init__()

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.net(x)

# Soll Modell gespeichert werden?
save_model = True

# Trainings- und Testdaten laden
features_training, labels_training, features_test, labels_test = extract_training_data('SimData__2025_03_27_09_36_50.mat')

# Daten vorbereiten 
scaler_f = StandardScaler()
scaler_l = StandardScaler()

scaled_features_training = scaler_f.fit_transform(features_training)
scaled_labels_training = scaler_l.fit_transform(labels_training)
scaled_features_test = scaler_f.transform(features_test)    # Hier nur transform, um Skalierungsparameter beizubehalten
scaled_labels_test = scaler_l.transform(labels_test)    # Hier nur transform, um Skalierungsparameter beizubehalten

# Trainings- und Testdaten in Torch-Tensoren umwandeln
features_tensor_training = torch.tensor(scaled_features_training, dtype=torch.float32)
labels_tensor_training = torch.tensor(scaled_labels_training, dtype=torch.float32)
features_tensor_test = torch.tensor(scaled_features_test, dtype=torch.float32)
labels_tensor_test = torch.tensor(scaled_labels_test, dtype=torch.float32)

# Dataset und Dataloader erstellen
dataset_training = TensorDataset(features_tensor_training, labels_tensor_training)
dataloader_training = DataLoader(dataset_training, batch_size=128, shuffle=True, drop_last=True, )
dataset_test = TensorDataset(features_tensor_test, labels_tensor_test)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, drop_last=False, )


# Neuronales Netz initialisieren
input_size = features_training.shape[1]
output_size = labels_training.shape[1]
hidden_size = 256

model = Feed_forward_NN(input_size, hidden_size, output_size)

# Loss funktionen und Optimierer wählen
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Optimierung (Lernprozess)
num_epochs = 50   # Anzahl der Durchläufe durch den gesamten Datensatz

print('Starte Optimierung...')

for epoch in range(num_epochs):
    # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
    model.train()
    loss_sum = 0

    for batch_features, batch_labels in dataloader_training:
        # Forward pass
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss des aktuellen Batches ausfsummieren
        loss_sum += loss.item()
    
    # Mittleren Loss berechnen und ausgeben
    training_loss_mean = loss_sum/len(dataloader_training)
   
    print(f'Epoch [{epoch+1}/{num_epochs}], Training-Loss: {training_loss_mean:.6f}')

# Modellvalidierung mit Testdaten
model.eval()
loss_sum = 0
with torch.no_grad():
    for batch_features, batch_labels in dataloader_test:
        # Forward Pass
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)

        # Loss des aktuellen Batches ausfsummieren
        loss_sum += loss.item()

# Mittleren Loss berechnen und ausgeben
test_loss_mean = loss_sum/len(dataloader_test)

print(f'Anwenden des trainierten Modells auf unbekannte Daten, Test-Loss: {test_loss_mean:.6f}')

if save_model == True:
    # Dummy Input für Export (gleiche Form wie deine Eingabedaten) - muss gemacht werden
    dummy_input = torch.randn(1, input_size)

    # Aktueller Zeitstempel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("Feedforward_NN", "Saved_Models", f"{timestamp}_feedforward_model.onnx")
    scaler_path = os.path.join("Feedforward_NN", "Saved_Models", f"{timestamp}_scaler.mat")

    # Modell exportieren
    torch.onnx.export(model, dummy_input, model_path, 
                    input_names=['input'], output_names=['output'], 
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                    opset_version=14)

    # Mittelwert und Std speichern
    scipy.io.savemat(scaler_path, {
        'mean_f': scaler_f.mean_,
        'scale_f': scaler_f.scale_,
        'mean_l': scaler_l.mean_,
        'scale_l': scaler_l.scale_
    })
