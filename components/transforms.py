import numpy as np

def clarke_transform(iu, iv, iw):
    ialpha = (2/3)*(iu - 0.5*iv - 0.5*iw)
    ibeta = (2/3)*(np.sqrt(3)/2)*(iv - iw)
    return ialpha, ibeta

def park_transform(ialpha, ibeta, theta):
    id_ = ialpha * np.cos(theta) + ibeta * np.sin(theta)
    iq = -ialpha * np.sin(theta) + ibeta * np.cos(theta)
    return id_, iq

def ipark_transform(vd, vq, theta):
    valpha = vd * np.cos(theta) - vq * np.sin(theta)
    vbeta = vd * np.sin(theta) + vq * np.cos(theta)
    return valpha, vbeta

def iclarke_transform(valpha, vbeta):
    vu = valpha
    vv = (-0.5) * valpha + (np.sqrt(3)/2) * vbeta
    vw = (-0.5) * valpha - (np.sqrt(3)/2) * vbeta
    return vu, vv, vw

def svm(valpha, vbeta, vdc):
    vu, vv, vw = iclarke_transform(valpha, vbeta)

    du = 0.5 + vu / vdc
    dv = 0.5 + vv / vdc
    dw = 0.5 + vw / vdc

    return du, dv, dw
