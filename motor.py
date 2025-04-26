import numpy as np
from scipy.integrate import solve_ivp

class BrushlessMotor:
    def __init__(self):
        self.R = 0.304       # Ohms (phase neutre)
        self.L = 0.2315e-3   # Henry
        self.kt = 0.0369     # Nm/A
        self.ke = 0.0369     # V.s/rad
        self.J = 2.4e-5      # kg.m^2
        self.pole_pairs = 4  # paires de pôles
        self.load_torque = 0 # Nm

    def electrical_back_emf(self, theta):
        ea = self.ke * np.sin(self.pole_pairs * theta)
        eb = self.ke * np.sin(self.pole_pairs * theta - 2*np.pi/3)
        ec = self.ke * np.sin(self.pole_pairs * theta + 2*np.pi/3)
        return ea, eb, ec

    def torque(self, ia, ib, ic, theta):
        fa = np.sin(self.pole_pairs * theta)
        fb = np.sin(self.pole_pairs * theta - 2*np.pi/3)
        fc = np.sin(self.pole_pairs * theta + 2*np.pi/3)
        return self.kt * (ia * fa + ib * fb + ic * fc)

    def dynamics(self, t, state, voltages):
        ia, ib, ic, omega, theta = state
        va, vb, vc = voltages
        ea, eb, ec = self.electrical_back_emf(theta)

        dia_dt = (va - self.R * ia - ea) / self.L
        dib_dt = (vb - self.R * ib - eb) / self.L
        dic_dt = (vc - self.R * ic - ec) / self.L

        Te = self.torque(ia, ib, ic, theta)
        domega_dt = (Te - self.load_torque) / self.J
        dtheta_dt = omega

        return [dia_dt, dib_dt, dic_dt, domega_dt, dtheta_dt]

    def simulate_step(self, state, voltages, dt):
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, voltages),
                        [0, dt], state, method='RK45', max_step=dt)
        return sol.y[:, -1]
