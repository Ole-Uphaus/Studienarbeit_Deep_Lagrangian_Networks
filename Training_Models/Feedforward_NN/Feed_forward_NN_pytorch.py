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
    features = data['features']
    labels = data['labels']

    return features, labels

# Erstellung neuronales Netz (Klasse)
class Feed_forward_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feed_forward_NN, self).__init__()

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.net(x)

# Soll Modell gespeichert werden?
save_model = True

# Trainingsdaten laden
features, labels = extract_training_data('SimData__2025_03_20_13_59_55.mat')

# Daten vorbereiten
scaler_f = StandardScaler()
scaler_l = StandardScaler()
scaled_features = scaler_f.fit_transform(features)
scaled_labels = scaler_l.fit_transform(labels)

# Trainingsdaten in Torch-Tensoren umwandeln
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
labels_tensor = torch.tensor(scaled_labels, dtype=torch.float32)

# Dataset und Dataloader erstellen
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, )

# Neuronales Netz initialisieren
input_size = features.shape[1]
output_size = labels.shape[1]
hidden_size = 256

model = Feed_forward_NN(input_size, hidden_size, output_size)

# Loss funktionen und Optimierer wählen
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Optimierung
num_epochs = 50   # Anzahl der Durchläufe durch den gesamten Datensatz

print('Starte Optimierung...')

for epoch in range(num_epochs):
    for batch_features, batch_labels in dataloader:
        # Forward pass
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
   
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Modell ausprobieren
model.eval()
with torch.no_grad():
    predictions = model(features_tensor[50000])

# In Numpy umwandeln und ausgeben
precictions = predictions.detach().numpy().reshape(1, -1)
print('Prädiktion:', scaler_l.inverse_transform(precictions))
print('Realität:', labels[50000])

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
