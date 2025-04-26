import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from motor import BrushlessMotor
from pwm import PWM
from adc import ADC
from transforms import clarke_transform, park_transform, inverse_park_transform
from pi_controller import PIController
from svm import svm

# Paramètres PWM
f_pwm = 20e3
T_pwm = 1 / f_pwm

# Suréchantillonnage
oversample_factor = 1
dt = T_pwm / oversample_factor

# Simulation
Tsim = 0.5
steps = int(Tsim / dt)

# Définition du moteur
motor = BrushlessMotor(
    R=0.304,
    L=0.2315e-3,
    kt=0.0369,
    ke=0.04,
    J=2.4e-5,
    pole_pairs=8,
    load_torque=0.0,
    friction_coefficient=0.015
)

pwm = PWM(vdc=48)
adc = ADC(bits=16, i_max=20)

id_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))
iq_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))

# Initialisation logs
logs = pd.DataFrame(columns=["time", "theta", "ialpha", "ibeta", "theta_elec", "theta_error"])

# Initialisation
theta_init = 187 / 180 * np.pi
state = np.array([0.0, 0.0, 0.0, 0.0, theta_init / motor.pole_pairs])  # [ia, ib, ic, omega, theta]

duty_a, duty_b, duty_c = 0.5, 0.5, 0.5

kpangle = 1.0
theta_initial = state[4]

integ = 0
for step in range(steps):
    t = step * dt

    theta_error = np.fmod(theta_initial - state[4], 2 * np.pi)
    integ += 0.01 * theta_error
    theta_elec = integ + 10.0 * theta_error

    iq_ref = min(t * 1.0 / Tsim * 2, 1.0)
    id_ref = 0.0

    # Application PWM -> tensions
    va, vb, vc = pwm.duty_to_voltage(duty_a, duty_b, duty_c)

    # Simulation moteur
    state = motor.simulate_step(state, (va, vb, vc), dt)
    ia, ib, ic, omega, theta = state

    # Mesure
    ia_adc, ib_adc, ic_adc = adc.sample(ia, ib, ic)

    # Transformation Clarke
    ialpha_meas, ibeta_meas = clarke_transform(ia_adc, ib_adc, ic_adc)

    # Simulation FOC en repère fixe
    id_meas, iq_meas = park_transform(ialpha_meas, ibeta_meas, theta_elec)

    vd = id_controller.compute(id_ref - id_meas)
    vq = iq_controller.compute(iq_ref - iq_meas)

    valpha, vbeta = inverse_park_transform(vd, vq, theta_elec)

    duty_a, duty_b, duty_c = svm(valpha, vbeta, pwm.vdc)

    # Logging
    logs.loc[step] = [t, theta / motor.pole_pairs, ialpha_meas, ibeta_meas, theta_elec, theta_error]

logs = logs.astype(float)

# Tracé Alpha/Beta
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(logs["time"], logs["ialpha"], label="i_alpha mesuré")
axs[0].plot(logs["time"], logs["ibeta"], label="i_beta mesuré")
axs[0].set_title('Courants Alpha-Beta')
axs[0].set_xlabel('Temps (s)')
axs[0].set_ylabel('Courant (A)')
axs[0].legend()
axs[0].grid()

axs[1].plot(logs["time"], logs["theta"]*180/np.pi, label="theta (mécanique)")
axs[1].plot(logs["time"], logs["theta_elec"]*180/np.pi - 90 , label="theta (interne)")
axs[1].plot(logs["time"], logs["theta_error"]*180/np.pi , label="theta (error)")
axs[1].set_title('Position Rotor')
axs[1].set_xlabel('Temps (s)')
axs[1].set_ylabel('Theta (deg)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
