import numpy as np

class PWM:
    def __init__(self, vdc):
        self.vdc = vdc  # Tension continue d'alimentation (ex: 48V)

    def duty_to_voltage(self, duty_a, duty_b, duty_c):
        va = (2 * duty_a - 1) * (self.vdc / 2)
        vb = (2 * duty_b - 1) * (self.vdc / 2)
        vc = (2 * duty_c - 1) * (self.vdc / 2)
        return va, vb, vc

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

    def sample(self, duty_a: float, duty_b: float, duty_c: float):
        """
        Retourne les tensions instantanées U/V/W (+Vdc/2 ou −Vdc/2) à l’instant courant,
        puis avance la porteuse d’un pas dt.
        """
        # consignes normalisées [0..1] -> valeurs comparateur [-1..+1]
        ma = 2*duty_a - 1
        mb = 2*duty_b - 1
        mc = 2*duty_c - 1

        va =  +self.vdc/2 if ma > self.carrier else -self.vdc/2
        vb =  +self.vdc/2 if mb > self.carrier else -self.vdc/2
        vc =  +self.vdc/2 if mc > self.carrier else -self.vdc/2

        self._update_carrier()
        return va, vb, vc
