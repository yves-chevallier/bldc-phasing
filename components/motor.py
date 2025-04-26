import numpy as np
from scipy.integrate import solve_ivp

class BrushlessMotor:
    def __init__(self, R, L, kt, ke, J, pole_pairs, load_torque=0.0, friction_coefficient=0.01):
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
        eu = self.ke * np.sin(self.pole_pairs * theta)
        ev = self.ke * np.sin(self.pole_pairs * theta - 2*np.pi/3)
        ew = self.ke * np.sin(self.pole_pairs * theta + 2*np.pi/3)
        return eu, ev, ew

    def torque(self, ia, ib, ic, theta):
        fa = np.sin(self.pole_pairs * theta)
        fb = np.sin(self.pole_pairs * theta - 2*np.pi/3)
        fc = np.sin(self.pole_pairs * theta + 2*np.pi/3)
        return self.kt * (ia * fa + ib * fb + ic * fc)

    def dynamics(self, t, state, voltages):
        ia, ib, ic, omega, theta = state
        vu, vv, vw = voltages
        eu, ev, ew = self.electrical_back_emf(theta)

        dia_dt = (vu - self.R * ia - eu) / self.L
        dib_dt = (vv - self.R * ib - ev) / self.L
        dic_dt = (vw - self.R * ic - ew) / self.L

        Te = self.torque(ia, ib, ic, theta)
        friction = self.friction_coefficient * omega
        domega_dt = (Te - self.load_torque - friction) / self.J
        dtheta_dt = omega

        return [dia_dt, dib_dt, dic_dt, domega_dt, dtheta_dt]

    def simulate_step(self, state, voltages, dt):
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, voltages),
                        [0, dt], state, method='RK45', max_step=dt)
        return sol.y[:, -1]

    def simulate_step_euler(self, state, voltages, dt):
        derivs = self.dynamics(0, state, voltages)  # t non utilisé
        next_state = state + dt * np.array(derivs)
        return next_state