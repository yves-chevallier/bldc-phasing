import numpy as np

class ADC:
    def __init__(self, bits=16, i_max=50):  # Courant maximum mesurable
        self.resolution = 2**bits
        self.i_max = i_max  # en Ampères
        self.buffer = None  # pour simuler le retard

    def sample(self, ia, ib, ic):
        # Simulation retard simple
        if self.buffer is None:
            self.buffer = (ia, ib, ic)

        measured_ia, measured_ib, measured_ic = self.buffer

        # Nouvelle valeur pour le prochain échantillonnage
        self.buffer = (ia, ib, ic)

        # Quantification
        measured_ia = np.clip(measured_ia, -self.i_max, self.i_max)
        measured_ib = np.clip(measured_ib, -self.i_max, self.i_max)
        measured_ic = np.clip(measured_ic, -self.i_max, self.i_max)

        measured_ia = np.round((measured_ia/self.i_max) * (self.resolution/2)) / (self.resolution/2) * self.i_max
        measured_ib = np.round((measured_ib/self.i_max) * (self.resolution/2)) / (self.resolution/2) * self.i_max
        measured_ic = np.round((measured_ic/self.i_max) * (self.resolution/2)) / (self.resolution/2) * self.i_max

        return measured_ia, measured_ib, measured_ic
