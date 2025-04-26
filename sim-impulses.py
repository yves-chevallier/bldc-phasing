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
T_pwm = 1 / f_pwm

# Facteur de suréchantillonnage pour la simulation
oversample_factor = 1
dt = T_pwm / oversample_factor

# Paramètres de simulation
Tsim = 0.08  # 8x50ms = 400ms
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

# Initialisation du DataFrame
logs = pd.DataFrame(columns=["time", "theta", "ialpha", "ialpha_ref", "ibeta", "ibeta_ref"])

# Initialisation du moteur
theta_init = 0 / 180 * np.pi  # Valeur arbitraire pour theta initial
state = np.array([0.0, 0.0, 0.0, 0.0, theta_init])  # [ia, ib, ic, omega, theta]

duty_a, duty_b, duty_c = 0.5, 0.5, 0.5

# Paramètres de la stimulation
thetas = np.array([0, 1, 0.5, -0.5, 0.25, -0.75, 0.75, -0.25]) * np.pi

pulse_width = 1e-3
pulses_interval = 9e-3
sequence_duration = pulse_width + pulses_interval
sequence_steps = int(sequence_duration / dt)

for step in range(steps):
    t = step * dt

    # Détermination de la séquence active
    sequence_index = (step // sequence_steps) % len(thetas)
    theta_elec = thetas[sequence_index]
    #theta_elec = np.pi / 4

    # Détermination si pulse actif
    step_in_sequence = step % sequence_steps
    if step_in_sequence * dt < pulse_width:
        iq_ref = 1.0
    else:
        iq_ref = 0.0

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

    # Consigne de courant en alpha/beta
    ialpha_ref, ibeta_ref = inverse_park_transform(id_ref, iq_ref, theta_elec)

    # Logging
    logs.loc[step] = [t, theta, ialpha_meas, ialpha_ref, ibeta_meas, ibeta_ref]

logs = logs.astype(float)

# Tracé
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(logs["time"], logs["ialpha"], label="i_alpha mesuré")
axs[0].plot(logs["time"], logs["ialpha_ref"], '--', label="i_alpha consigne")
axs[0].plot(logs["time"], logs["ibeta"], label="i_beta mesuré")
axs[0].plot(logs["time"], logs["ibeta_ref"], '--', label="i_beta consigne")
axs[0].set_title('Courants Alpha-Beta')
axs[0].set_xlabel('Temps (s)')
axs[0].set_ylabel('Courant (A)')
axs[0].legend()
axs[0].grid()

axs[1].plot(logs["time"], logs["theta"], label="theta (mécanique)")
axs[1].set_title('Position Rotor')
axs[1].set_xlabel('Temps (s)')
axs[1].set_ylabel('Theta (rad)')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
