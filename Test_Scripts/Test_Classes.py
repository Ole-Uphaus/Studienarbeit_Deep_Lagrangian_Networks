import torch
import numpy as np

class Fahrzeug:
    def __init__(self, marke):
        self.marke = marke
        print(f"Fahrzeug der Marke {self.marke} wurde erstellt.")

    def info(self):
        print(f"Die Marke des Fahrzeuges lautet {self.marke}")

class Auto(Fahrzeug):
    def __init__(self, marke, modell):
        # Hier wird der Konstruktor der Elternklasse aufgerufen
        super(Auto, self).__init__(marke)
        self.modell = modell
        print(f"Auto-Modell {self.modell} wurde erstellt.")

    def info(self):
        print(f"Das modell des Fahrzeugs lautet {self.modell}")


bmw = Fahrzeug("BMW")
bmw.info()

vw = Auto(marke="VW", modell="Golf")
vw.info()

print(np.zeros(4))
