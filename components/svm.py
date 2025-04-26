import numpy as np

def svm(valpha, vbeta, vdc):
    """
    Génère des rapports cycliques simples pour le PWM.
    """
    # Transforme alpha-beta en abc simple
    va = valpha
    vb = -0.5 * valpha + (np.sqrt(3)/2) * vbeta
    vc = -0.5 * valpha - (np.sqrt(3)/2) * vbeta

    # Normalisation par Vdc
    d_a = 0.5 + va / vdc
    d_b = 0.5 + vb / vdc
    d_c = 0.5 + vc / vdc

    return d_a, d_b, d_c
