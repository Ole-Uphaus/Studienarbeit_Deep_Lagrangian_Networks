# Studienarbeit: Deep Lagrangian Networks

In diesem Repository befindet sich der gesamte Code, der im Rahmen meiner Studienarbeit zu **Deep Lagrangian Networks (DeLaN)** entstanden ist.  
Ziel der Arbeit ist die Modellierung und Validierung dynamischer Systeme mithilfe von Deep-Learning-Ansätzen, die die Lagrange-Mechanik berücksichtigen.

---

## Repository-Struktur

- **Data_generation/**  
  Skripte zur Datenerzeugung für das Training und die Validierung der Modelle.  
  Enthält u.a. Simulationsskripte sowie Tools zur Aufbereitung der Rohdaten.

- **Mujoco/**  
  Dateien und Skripte zur Simulation in [MuJoCo](https://mujoco.org/).  
  Hier werden physikalische Modelle erzeugt, die als Grundlage für Trainings- und Testdaten dienen.

- **Test_Scripts/**  
  Verschiedene Testskripte für einzelne Module, Trainingseinstellungen und Validierungsroutinen.

- **Training_Data/**  
  Enthält vorbereitete Datensätze, die für das Training und die Evaluierung verwendet werden.

- **Training_Models/**  
  Gespeicherte Modelle und Checkpoints der trainierten Deep Lagrangian Networks.  
  Dient der Reproduzierbarkeit und weiteren Analyse.

- **Validation_DeLaN_Lutter.py**  
  Python-Skript zur Validierung des DeLaN-Ansatzes nach der Originalarbeit von Lutter et al.

- **Validation_FFNN.py**  
  Python-Skript zur Validierung eines Feed-Forward Neural Networks als Vergleichsmodell.

- **Validation_FFNN.m**  
  MATLAB-Skript für die Validierung des FFNN in MATLAB-Umgebung.

---

## Voraussetzungen

- **Python 3.9+**  
  Empfohlene Bibliotheken:  
  - `torch`  
  - `numpy`  
  - `matplotlib`  
  - ggf. `mujoco-py`

- **MATLAB R2022a+**  
  Für die Validierungsskripte in MATLAB.

---

## Nutzung

1. Datengenerierung starten (`Data_generation/`).  
2. Trainingsskripte ausführen, Modelle werden in `Training_Models/` gespeichert.  
3. Validierung der Modelle mit den entsprechenden Skripten (`Validation_DeLaN_Lutter.py`, `Validation_FFNN.*`).  

---

## Hinweise

- Dieses Repository ist im Rahmen einer Studienarbeit entstanden und nicht als fertiges Softwarepaket gedacht.  
- Für Reproduzierbarkeit und weitere Analysen wurden alle Trainingsskripte, Daten und Modelle beigelegt.  
- Bei Fragen oder Feedback bitte gerne ein Issue eröffnen.

---
