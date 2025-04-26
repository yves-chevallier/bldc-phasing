import numpy as np

class ADC:
    def __init__(self, bits=16, i_max=50):
        self.resolution = 2**bits
        self.i_max = i_max
        self.buffer = np.zeros(3)

    def sample(self, iu, iv, iw):
        currents_in = np.array([iu, iv, iw])
        i_meas = self.buffer
        self.buffer = currents_in

        i_meas = np.clip(i_meas, -self.i_max, self.i_max) / self.i_max
        quantized = np.round(i_meas * (self.resolution / 2)) / (self.resolution / 2)

        return quantized * self.i_max
