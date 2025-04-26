import numpy as np

def clarke_transform(ia, ib, ic):
    ialpha = (2/3)*(ia - 0.5*ib - 0.5*ic)
    ibeta = (2/3)*(np.sqrt(3)/2)*(ib - ic)
    return ialpha, ibeta

def park_transform(ialpha, ibeta, theta):
    id_ = ialpha * np.cos(theta) + ibeta * np.sin(theta)
    iq = -ialpha * np.sin(theta) + ibeta * np.cos(theta)
    return id_, iq

def inverse_park_transform(vd, vq, theta):
    valpha = vd * np.cos(theta) - vq * np.sin(theta)
    vbeta = vd * np.sin(theta) + vq * np.cos(theta)
    return valpha, vbeta
