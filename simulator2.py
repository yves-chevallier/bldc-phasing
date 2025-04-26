import numpy as np
import matplotlib.pyplot as plt

from motor import BrushlessMotor
from pwm import PWM
from adc import ADC
from transforms import clarke_transform, park_transform, inverse_park_transform
from pi_controller import PIController
from svm import svm

# Paramètres de simulation
dt = 50e-6
Tsim = 0.5
steps = int(Tsim / dt)

# Instanciation des modules
motor = BrushlessMotor()
pwm = PWM(vdc=48)
adc = ADC(bits=16, i_max=20)

id_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))
iq_controller = PIController(kp=2.0, ki=1500.0, dt=dt, output_limits=(-24, 24))

# État initial du moteur
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [ia, ib, ic, omega, theta]

# Pour stockage des résultats
time_log = []
omega_log = []
theta_log = []
ia_log = []
ib_log = []
ic_log = []
id_log = []
iq_log = []
iq_ref_log = []

# Initialisation PWM
duty_a = 0.5
duty_b = 0.5
duty_c = 0.5

# Boucle de simulation
for step in range(steps):
    t = step * dt

    # Génération d'une consigne carrée iq à 100 Hz
    # if np.sin(2 * np.pi * 100 * t) >= 0:
    #     iq_ref = 1.0
    # else:
    #     iq_ref = -1.0
    iq_ref = 1.0
    id_ref = 0

    # PWM --> Tensions appliquées aux phases
    va, vb, vc = pwm.duty_to_voltage(duty_a, duty_b, duty_c)

    # Moteur --> Simulation dynamique
    state = motor.simulate_step(state, (va, vb, vc), dt)
    ia, ib, ic, omega, theta = state

    # ADC --> Mesure courants avec retard et quantification
    ia_adc, ib_adc, ic_adc = adc.sample(ia, ib, ic)

    # Transformations Clarke & Park
    ialpha, ibeta = clarke_transform(ia_adc, ib_adc, ic_adc)
    id_meas, iq_meas = park_transform(ialpha, ibeta, theta)

    # Contrôleurs PI sur id et iq
    vd = id_controller.compute(0.0 - id_meas)  # id_ref = 0
    vq = iq_controller.compute(iq_ref - iq_meas)

    # Inverse Park pour générer v_alpha, v_beta
    valpha, vbeta = inverse_park_transform(vd, vq, theta)

    # Modulateur SVM --> Duty cycles
    duty_a, duty_b, duty_c = svm(valpha, vbeta, pwm.vdc)

    # Enregistrement des résultats
    time_log.append(t)
    omega_log.append(omega)
    theta_log.append(theta)
    ia_log.append(ia)
    ib_log.append(ib)
    ic_log.append(ic)
    id_log.append(id_meas)
    iq_log.append(iq_meas)
    iq_ref_log.append(iq_ref)

# Transformation des logs en tableaux numpy
time_log = np.array(time_log)
omega_log = np.array(omega_log)
theta_log = np.array(theta_log)
ia_log = np.array(ia_log)
ib_log = np.array(ib_log)
ic_log = np.array(ic_log)
id_log = np.array(id_log)
iq_log = np.array(iq_log)
iq_ref_log = np.array(iq_ref_log)

plt.figure()
plt.plot(time_log, omega_log * 60 / (2*np.pi))  # Convertir rad/s en tr/min
plt.title('Vitesse Rotor (tr/min)')
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (tr/min)')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time_log, ia_log, label="ia")
plt.plot(time_log, ib_log, label="ib")
plt.plot(time_log, ic_log, label="ic")
plt.title('Courants de phase')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time_log, id_log, label="id")
plt.plot(time_log, iq_log, label="iq")
plt.title('Courants en d/q')
plt.xlabel('Temps (s)')
plt.ylabel('Courant (A)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(time_log, iq_ref_log, label="iq_ref", linestyle='--')
plt.plot(time_log, iq_log, label="iq_meas")
plt.title('Comparaison iq_ref / iq_meas')
plt.xlabel('Temps (s)')
plt.ylabel('Courant q (A)')
plt.legend()
plt.grid(True)
plt.show()
