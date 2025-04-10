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
from pathlib import Path


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
    def __init__(self, input_size, hidden_size, output_size, dropout, depth):
        super(Feed_forward_NN, self).__init__()

        layers = []

        # Input Schicht
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden Layers als Schleife anhängen
        for i in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output Layer hinzufügen
        layers.append(nn.Linear(hidden_size, output_size))

        # Hier eine Vereinfachung, dass der befehl in forward Funktion kürzer wird
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
# Bedingung, damit beim laden des Files nicht der ganze code ausgeführt wird, sondern nur die klasse genommen wird   
if __name__ == "__main__":

    # Checken, ob Cuda verfügbar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benutze Gerät: {device}")

    # Trainings- und Testdaten laden
    features_training, labels_training, features_test, labels_test = extract_training_data('SimData__2025_04_04_09_51_52.mat')

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

    # Parameter festlegen
    hyper_param = {'save_model': True,
                'epoch': 100,
                'hidden_size': 256,
                'batch_size': 512,
                'learning_rate': 0.001,
                'wheight_decay': 1e-5,
                'dropout': 0.3,
                'input_size': features_training.shape[1],
                'output_size': labels_training.shape[1],
                'depth': 2}

    # Neuronales Netz initialisieren
    model = Feed_forward_NN(hyper_param['input_size'], hyper_param['hidden_size'], hyper_param['output_size'], hyper_param['dropout'], hyper_param['depth']).to(device)

    # Loss funktionen und Optimierer wählen
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_param['learning_rate'], weight_decay=hyper_param['wheight_decay'])

    # Dataset und Dataloader erstellen
    dataset_training = TensorDataset(features_tensor_training, labels_tensor_training)
    dataloader_training = DataLoader(dataset_training, batch_size=hyper_param['batch_size'], shuffle=True, drop_last=True, )
    dataset_test = TensorDataset(features_tensor_test, labels_tensor_test)
    dataloader_test = DataLoader(dataset_test, batch_size=hyper_param['batch_size'], shuffle=False, drop_last=False, )

    # Optimierung (Lernprozess)
    num_epochs = hyper_param['epoch']  # Anzahl der Durchläufe durch den gesamten Datensatz

    print('Starte Optimierung...')

    for epoch in range(num_epochs):
        # Modell in den Trainingsmodeus versetzen und loss Summe initialisieren
        model.train()
        loss_sum = 0

        for batch_features, batch_labels in dataloader_training:

            # Tensoren auf GPU schieben
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

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

            # Tensoren auf GPU schieben
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward Pass
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)

            # Loss des aktuellen Batches ausfsummieren
            loss_sum += loss.item()

    # Mittleren Loss berechnen und ausgeben
    test_loss_mean = loss_sum/len(dataloader_test)

    print(f'Anwenden des trainierten Modells auf unbekannte Daten, Test-Loss: {test_loss_mean:.6f}')

    if hyper_param['save_model'] == True:
        # Dummy Input für Export (gleiche Form wie deine Eingabedaten) - muss gemacht werden
        dummy_input = torch.randn(1, hyper_param['input_size']).to(device)

        # Aktueller Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = Path(__file__).resolve()
        model_path = os.path.join(script_path.parents[0], "Saved_Models", f"{timestamp}_feedforward_model.onnx")
        scaler_path = os.path.join(script_path.parents[0], "Saved_Models", f"{timestamp}_scaler.mat")

        # Modell exportieren (MATLAB)
        torch.onnx.export(model, dummy_input, model_path, 
                        input_names=['input'], output_names=['output'], 
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                        opset_version=14)
        
        # Modell exportieren (Python)
        model_path = os.path.join(script_path.parents[0], "Saved_Models", f"{timestamp}_feedforward_model.pth")

        torch.save({
            'state_dict': model.state_dict(),
            'input_size': hyper_param['input_size'],
            'output_size': hyper_param['output_size'],
            'hidden_size': hyper_param['hidden_size'],
            'dropout': hyper_param['dropout'],
            'depth': hyper_param['depth'],
            'scaler_f': scaler_f,
            'scaler_l': scaler_l,
        }, model_path)

        # Mittelwert und Std speichern
        scipy.io.savemat(scaler_path, {
            'mean_f': scaler_f.mean_,
            'scale_f': scaler_f.scale_,
            'mean_l': scaler_l.mean_,
            'scale_l': scaler_l.scale_
        })
