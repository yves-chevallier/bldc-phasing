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
f_pwm = 20e3
T_pwm = 1 / f_pwm

# Suréchantillonnage
oversample_factor = 1
dt = T_pwm / oversample_factor

thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
pulse_width = 1e-3
pulses_interval = 9e-3

# Simulation
Tsim = (pulse_width + pulses_interval) * len(thetas)
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
logs = pd.DataFrame(columns=["time", "theta", "ialpha", "ibeta"])

# Initialisation
theta_init = 180 / 180 * np.pi
state = np.array([0.0, 0.0, 0.0, 0.0, theta_init / motor.pole_pairs])  # [ia, ib, ic, omega, theta]

dc = np.ones(3) * 0.5

#thetas = np.array([0, 1, 0.5, -0.5, 0.25, -0.75, 0.75, -0.25]) * np.pi

#thetas = np.array([0, 1, 0.5, -0.5, 0.25, -0.75, 0.75, -0.25]) * np.pi + theta_init



sequence_duration = pulse_width + pulses_interval
sequence_steps = int(sequence_duration / dt)

# Pour stockage du delta position
delta_thetas = []
excitation_angles = []
theta_real_start = []

recording = False
theta_start = 0

for step in range(steps):
    t = step * dt

    sequence_index = (step // sequence_steps) % len(thetas)
    theta_elec = thetas[sequence_index]

    step_in_sequence = step % sequence_steps
    pulse_active = (step_in_sequence * dt) < pulse_width

    if pulse_active:
        if not recording:
            # Début de pulse
            theta_start = state[4]
            recording = True
    else:
        if recording:
            # Fin de pulse
            theta_end = state[4]
            delta_theta = theta_end - theta_start
            delta_thetas.append(delta_theta)
            excitation_angles.append(theta_elec)
            theta_real_start.append(theta_start)
            recording = False

    iq_ref = 1.0 if pulse_active else 0.0
    id_ref = 0.0

    # Application PWM -> tensions
    vuvw = pwm.duty_to_voltage(dc)

    # Simulation moteur
    state = motor.simulate_step_euler(state, vuvw, dt)
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

    dc = np.array(svm(valpha, vbeta, pwm.vdc))

    # Logging
    logs.loc[step] = [t, theta, ialpha_meas, ibeta_meas]

logs = logs.astype(float)

# Tracé Alpha/Beta
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# axs[0].plot(logs["time"], logs["ialpha"], label="i_alpha mesuré")
# axs[0].plot(logs["time"], logs["ibeta"], label="i_beta mesuré")
# axs[0].set_title('Courants Alpha-Beta')
# axs[0].set_xlabel('Temps (s)')
# axs[0].set_ylabel('Courant (A)')
# axs[0].legend()
# axs[0].grid()

# axs[1].plot(logs["time"], logs["theta"], label="theta (mécanique)")
# axs[1].set_title('Position Rotor')
# axs[1].set_xlabel('Temps (s)')
# axs[1].set_ylabel('Theta (rad)')
# axs[1].legend()
# axs[1].grid()

# plt.tight_layout()
# plt.show()

# Tracé polaire
delta_thetas = np.array(delta_thetas)
excitation_angles = np.array(excitation_angles)
theta_real_start = np.array(theta_real_start) * motor.pole_pairs

# Normalisation des amplitudes
delta_amplitude = delta_thetas
delta_amplitude /= np.max(np.abs(delta_amplitude))

# Composantes cartésiennes
dx = delta_amplitude * np.cos(excitation_angles)
dy = delta_amplitude * np.sin(excitation_angles)

# Tracé
plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

# Flèche verte pour la position réelle
ax.quiver(theta_real_start[0], 0, 0, 1,
          angles='xy', scale_units='xy', scale=1, color='g', label="Real position", alpha=0.5)

for i in range(len(excitation_angles)):
    color = 'b' if delta_amplitude[i] >= 0 else 'r'
    ax.quiver(excitation_angles[i], 0, 0, np.abs(delta_amplitude[i]),
              angles='xy', scale_units='xy', scale=1, color=color, alpha=0.5)

ax.set_ylim(0, 1.1)

ax.set_title('Réponse polaire impulsions')
plt.legend()
plt.show()

# Moyenne angulaire pondérée
corr_angles = np.where(delta_thetas >= 0, excitation_angles, excitation_angles + np.pi)

# 2) construction du vecteur pondéré
R = (np.abs(delta_thetas) * np.exp(1j * corr_angles)).sum()

# 3) estimation (avec rotation de 180° pour tomber sur la branche dominante)
theta_est = (np.angle(-R) + 2*np.pi) % (2*np.pi)
print(f"θ estimé = {np.degrees(theta_est):.1f}°")
