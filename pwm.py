class PWM:
    def __init__(self, vdc):
        self.vdc = vdc  # Tension continue d'alimentation (ex: 48V)

    def duty_to_voltage(self, duty_a, duty_b, duty_c):
        va = (2 * duty_a - 1) * (self.vdc / 2)
        vb = (2 * duty_b - 1) * (self.vdc / 2)
        vc = (2 * duty_c - 1) * (self.vdc / 2)
        return va, vb, vc
