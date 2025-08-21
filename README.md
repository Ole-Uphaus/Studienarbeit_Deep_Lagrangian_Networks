# Studienarbeit: Deep Lagrangian Networks

In diesem Repository befindet sich der gesamte Code, der im Rahmen meiner Studienarbeit zu **Deep Lagrangian Networks (DeLaN)** entstanden ist. 

---

## Repository-Struktur

- **Data_generation/**  
  Skripte zur Datenerzeugung für das Training und die Validierung der Modelle.  
  Enthält vor allem MATLAB Skripte und Funktionen.

- **Mujoco/**  
  Modelldateien und Skripte zur Simulation und Regelung.  

- **Test_Scripts/**  
  Kleine Testskripte - meist Jupyter Notebooks.

- **Training_Data/**  
  Enthält vorbereitete Datensätze, die für das Training und die Evaluierung von DeLaN verwendet werden.

- **Training_Models/**  
  Hier befinden sich die Wichtigsten Skripte, die das DeLaN Modell, sowie Funktionen und Trainingsskripte enthalten.
  Alle Experimentellen Untersuchungen meiner Studienarbeit sind in einem separametn Unterordner abgelegt.

---

## Nutzung

1. Trainingsdaten können mit einem der Skripte aus (`Data_generation/`) auf verschiedene weisen für verschiedene Systemmodelle generiert werden.  
2. Hauptskript ist das (`Data_generation/DeLaN_Ole/DeLaN_training_Ole.py`) Skript, mit dem sich das DeLaN Modell auf beliebigen Trainingsdaten mit variablen Hyperparametern trainieren lässt.
3. Die Regelungsimplementierungen befindet sich unter (`Mujoco/Control`). Es kann das jeweils vortrainierte Systemmodell kann in den entsprechenden Skripten ausgewählt werden.
