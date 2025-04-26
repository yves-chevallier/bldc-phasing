import numpy as np
from numpy.typing import NDArray

class PWM:
    def __init__(self, vdc):
        self.vdc = vdc

    def duty_to_voltage(self, dc: NDArray[np.float64]) -> NDArray[np.float64]:
        return (2.0 * dc - 1.0) * (self.vdc / 2.0)

class CenterAlignedPWM:
    """
    PWM up-down (centre-aligné) multiniveau 2.
    On met à jour le portaillage à chaque appel de .sample().
    """
    def __init__(self, vdc: float, fpwm: float, dt_sim: float):
        self.vdc = vdc
        self.Tc  = 1.0 / fpwm          # période PWM
        self.dt  = dt_sim              # pas de simulation
        self.carrier = -1.0            # démarre en fond de dent de scie
        self.dir = +1                  # +1 = monte, -1 = descend

    def _update_carrier(self):
        """Pas de simulation pour la porteuse triangulaire centre-alignée."""
        self.carrier += self.dir * 2 * self.dt / self.Tc
        if self.carrier >= +1.0:   # sommet : demi-tour
            self.carrier = +1.0
            self.dir = -1
        elif self.carrier <= -1.0: # fond : demi-tour
            self.carrier = -1.0
            self.dir = +1

    def sample(self, dc_u: float, dc_v: float, dc_w: float):
        """
        Retourne les tensions instantanées U/V/W (+Vdc/2 ou -Vdc/2) à l'instant courant,
        puis avance la porteuse d'un pas dt.
        """
        # consignes normalisées [0..1] -> valeurs comparateur [-1..+1]
        mu = 2*dc_u - 1
        mv = 2*dc_v - 1
        mw = 2*dc_w - 1

        vu =  +self.vdc/2 if mu > self.carrier else -self.vdc/2
        vv =  +self.vdc/2 if mv > self.carrier else -self.vdc/2
        vw =  +self.vdc/2 if mw > self.carrier else -self.vdc/2

        self._update_carrier()
        return vu, vv, vw
