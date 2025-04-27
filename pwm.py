import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from components.motor import BrushlessMotor
from components.pwm import CenterAlignedPWM
from components.adc import ADC
from components.transforms import clarke_transform, park_transform, ipark_transform, svm
from components.pi_controller import PIController

# Reprendre vos classes CenterAlignedPWM, BrushlessMotor, etc.

# Paramètres PWM
f_pwm = 50e3  # 20 kHz
T_pwm = 1 / f_pwm  # 50 µs

# Suréchantillonnage très élevé
oversample_factor = 200  # beaucoup plus fin
dt = T_pwm / oversample_factor

# Paramètres de simulation
Tsim = 0.003  # 3 ms suffisent pour voir la montée initiale
steps = int(Tsim / dt)

# Initialisation
motor = BrushlessMotor(
    R=0.600,
    L=0.500e-3,
    kt=0.0369,
    ke=0.04,
    J=2.4e-5,
    pole_pairs=8,
    load_torque=0.0
)

pwm = CenterAlignedPWM(vdc=24, fpwm=f_pwm, dt_sim=dt)
adc = ADC(bits=16, i_max=20)

id_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))
iq_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))

logs = pd.DataFrame(columns=["time", "omega", "theta", "ia", "ib", "ic", "id", "iq", "iq_ref", "duty_u", "duty_v", "duty_w", "vu", "vv", "vw"])

state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [ia, ib, ic, omega, theta]

# Démarrage avec des rapports cycliques à 50%
du, dv, dw = 0.5, 0.5, 0.5

for step in range(steps):
    t = step * dt

    iq_ref, id_ref = 10.0, 0.0  # 1A de consigne q, 0A en d

    # Générer tension instantanée U/V/W selon PWM centre-aligné
    vuvw = pwm.sample(du, dv, dw)

    # Simulation dynamique moteur à ce pas
    state = motor.simulate_step_euler(state, vuvw, dt)
    ia, ib, ic, omega, theta = state

    # Acquisition via ADC
    ia_adc, ib_adc, ic_adc = adc.sample(ia, ib, ic)

    # Transformation dans d/q
    theta_elec = motor.pole_pairs * theta
    ialpha, ibeta = clarke_transform(ia_adc, ib_adc, ic_adc)
    id_meas, iq_meas = park_transform(ialpha, ibeta, theta_elec)

    # Régulation PI
    vd = id_controller.compute(id_ref - id_meas)
    vq = iq_controller.compute(iq_ref - iq_meas)

    # Retour en αβ
    valpha, vbeta = ipark_transform(vd, vq, theta_elec)

    # Calcul nouveaux rapports cycliques
    du, dv, dw = svm(valpha, vbeta, pwm.vdc)

    # Stockage
    logs.loc[step] = [t, omega, theta, ia, ib, ic, id_meas, iq_meas, iq_ref, du, dv, dw, vuvw[0], vuvw[1], vuvw[2]]

logs = logs.astype(float)

plt.figure()
plt.plot(logs["time"], logs["ia"], label="ia")
plt.plot(logs["time"], logs["ib"], label="ib")
plt.plot(logs["time"], logs["ic"], label="ic")
plt.title('Courants de phase - avec ondulations PWM')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(logs["time"], logs["id"], label="id")
plt.plot(logs["time"], logs["iq"], label="iq")
plt.title('Courants d/q régulés')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(logs["time"], logs["vu"], label="Vu")
plt.plot(logs["time"], logs["vv"], label="Vv")
plt.plot(logs["time"], logs["vw"], label="Vw")
plt.plot(logs["time"], logs["ia"], label="ia")
plt.plot(logs["time"], logs["ib"], label="ib")
plt.plot(logs["time"], logs["ic"], label="ic")
plt.plot(logs["time"], logs["id"], label="id")
plt.plot(logs["time"], logs["iq"], label="iq")
plt.title('Tensions de phase - avec ondulations PWM')
plt.xlabel('Temps (s)')
plt.ylabel('Tension (V)')
plt.legend()
plt.grid(True)
plt.show()
