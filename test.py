import numpy as np
from scipy.integrate import solve_ivp

def clarke_transform(ia, ib, ic):
    ialpha = (2/3)*(ia - 0.5*ib - 0.5*ic)
    ibeta = (2/3)*(np.sqrt(3)/2)*(ib - ic)
    return ialpha, ibeta

def park_transform(ialpha, ibeta, theta):
    id_ = ialpha * np.cos(theta) + ibeta * np.sin(theta)
    iq = -ialpha * np.sin(theta) + ibeta * np.cos(theta)
    return id_, iq

def inverse_park_transform(vd, vq, theta):
    valpha = vd * np.cos(theta) - vq * np.sin(theta)
    vbeta = vd * np.sin(theta) + vq * np.cos(theta)
    return valpha, vbeta


class BrushlessMotor:
    def __init__(self):
        # Paramètres moteurs
        self.R = 0.304        # Ohms
        self.L = 0.2315e-3    # Henry
        self.kt = 0.0369      # Nm/A
        self.ke = 0.0369      # V.s/rad
        self.J = 2.4e-5       # kg.m^2
        self.pair_poles = 4   # paires de pôles
        self.Tl = 0           # Couple résistant supposé nul initialement

    def electrical_back_emf(self, theta):
        # Fonctions sinusoïdales (modèle Brushless sinus)
        ea = self.ke * np.sin(theta * self.pair_poles)
        eb = self.ke * np.sin(theta * self.pair_poles - 2*np.pi/3)
        ec = self.ke * np.sin(theta * self.pair_poles + 2*np.pi/3)
        return ea, eb, ec

    def torque(self, ia, ib, ic, theta):
        # Calcul du couple
        fa = np.sin(theta * self.pair_poles)
        fb = np.sin(theta * self.pair_poles - 2*np.pi/3)
        fc = np.sin(theta * self.pair_poles + 2*np.pi/3)
        return self.kt * (ia * fa + ib * fb + ic * fc)

    def dynamics(self, t, state, voltages):
        ia, ib, ic, omega, theta = state
        va, vb, vc = voltages

        ea, eb, ec = self.electrical_back_emf(theta)

        # Équations électriques
        dia_dt = (va - self.R * ia - ea) / self.L
        dib_dt = (vb - self.R * ib - eb) / self.L
        dic_dt = (vc - self.R * ic - ec) / self.L

        # Équation mécanique
        Te = self.torque(ia, ib, ic, theta)
        domega_dt = (Te - self.Tl) / self.J

        dtheta_dt = omega

        return [dia_dt, dib_dt, dic_dt, domega_dt, dtheta_dt]

    def simulate_step(self, state, voltages, dt):
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, voltages),
                        [0, dt], state, method='RK45', max_step=dt)
        return sol.y[:, -1]  # Dernier état

# Exemple d'utilisation
motor = BrushlessMotor()

# Etat initial : ia, ib, ic, omega, theta
state = [0, 0, 0, 0, 0]

# Entrées de tension
voltages = [10, -5, -5]  # U, V, W en Volts

# Simulation sur 1 ms
dt = 1e-3
new_state = motor.simulate_step(state, voltages, dt)

print("Nouveau state :", new_state)
