import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from components.motor import BrushlessMotor
from components.pwm import PWM
from components.adc import ADC
from components.transforms import clarke_transform, park_transform, inverse_park_transform
from components.pi_controller import PIController
from components.svm import svm

# Paramètres PWM
f_pwm = 20e3  # 20 kHz
T_pwm = 1 / f_pwm  # 50 us

# Facteur de suréchantillonnage pour la simulation
oversample_factor = 5  # 5 points par période PWM
dt = T_pwm / oversample_factor

# Paramètres de simulation
Tsim = 0.1  # Durée totale de simulation
steps = int(Tsim / dt)

motor = BrushlessMotor(
    R=0.304, # Ohm
    L=0.2315e-3, # Henri
    kt=0.0369, # Nm/A
    ke=0.04, # V.s/rad
    J=2.4e-5, # kg.m^2
    pole_pairs=8,
    load_torque=0.01
)

pwm = PWM(vdc=48)
adc = ADC(bits=16, i_max=20)

id_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))
iq_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))

logs = pd.DataFrame(columns=["time", "omega", "theta", "ia", "ib", "ic", "id", "iq", "iq_ref"])

state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [ia, ib, ic, omega, theta]

duty_a, duty_b, duty_c = 0.5, 0.5, 0.5

for step in range(steps):
    t = step * dt

    iq_ref, id_ref = 1.0, 0.0

    va, vb, vc = pwm.duty_to_voltage(duty_a, duty_b, duty_c)

    state = motor.simulate_step(state, (va, vb, vc), dt)
    ia, ib, ic, omega, theta = state

    ia_adc, ib_adc, ic_adc = adc.sample(ia, ib, ic)

    theta_elec = motor.pole_pairs * theta
    ialpha, ibeta = clarke_transform(ia_adc, ib_adc, ic_adc)
    id_meas, iq_meas = park_transform(ialpha, ibeta, theta_elec)

    vd = id_controller.compute(id_ref - id_meas)
    vq = iq_controller.compute(iq_ref - iq_meas)

    valpha, vbeta = inverse_park_transform(vd, vq, theta_elec)

    duty_a, duty_b, duty_c = svm(valpha, vbeta, pwm.vdc)

    logs.loc[step] = [t, omega, theta, ia, ib, ic, id_meas, iq_meas, iq_ref]

logs = logs.astype(float)

plt.figure()
plt.plot(logs["time"], logs["omega"] * 60 / (2*np.pi))
plt.title('Vitesse Rotor (tr/min)')
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (tr/min)')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(logs["time"], logs["ia"], label="ia")
plt.plot(logs["time"], logs["ib"], label="ib")
plt.plot(logs["time"], logs["ic"], label="ic")
plt.title('Courants de phase')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(logs["time"], logs["id"], label="id")
plt.plot(logs["time"], logs["iq"], label="iq")
plt.title('Courants en d/q')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(logs["time"], logs["iq_ref"], label="iq_ref", linestyle='--')
plt.plot(logs["time"], logs["iq"], label="iq_meas")
plt.title('Comparaison iq_ref / iq_meas')
plt.xlabel('Temps (s)')
plt.ylabel('Courant q (A)')
plt.legend()
plt.grid(True)
plt.show()

# (Optionnel) Sauvegarder les résultats
# logs.to_csv("simulation_results.csv", index=False)
