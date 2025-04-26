import numpy as np
from scipy.integrate import solve_ivp

class BrushlessMotor:
    def __init__(self, R, L, kt, ke, J, pole_pairs, load_torque=0.0, friction_coefficient=0.0):
        """
        Initialize a Brushless Motor model.

        Parameters:
        R : float
            Phase-to-neutral resistance (Ohms)
        L : float
            Phase-to-neutral inductance (Henrys)
        kt : float
            Torque constant (Nm/A)
        ke : float
            Back-EMF constant (V.s/rad)
        J : float
            Rotor inertia (kg.m^2)
        pole_pairs : int
            Number of pole pairs
        load_torque : float
            Constant external load torque (Nm)
        friction_coefficient : float
            Viscous friction coefficient (Nm.s/rad)
        """
        self.R = R
        self.L = L
        self.kt = kt
        self.ke = ke
        self.J = J
        self.pole_pairs = pole_pairs
        self.load_torque = load_torque
        self.friction_coefficient = friction_coefficient

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
        friction = self.friction_coefficient * omega
        domega_dt = (Te - self.load_torque - friction) / self.J
        dtheta_dt = omega

        return [dia_dt, dib_dt, dic_dt, domega_dt, dtheta_dt]

    def simulate_step(self, state, voltages, dt):
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, voltages),
                        [0, dt], state, method='RK45', max_step=dt)
        return sol.y[:, -1]
