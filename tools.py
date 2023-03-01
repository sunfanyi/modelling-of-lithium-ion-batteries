import numpy as np


def match_val(target, x, y):
    """
    Match the corresponding y value of a target value, where y = f(x).
    np.interp() is not used because it's computationally expensive and unstable
    for large arrays.

    eg., to find OCV(z=1), use match_val(1, z, OCV)
    """
    idx = np.abs(x - target).argmin()
    res = y[idx]

    return res


def update_SOC(i, z, t, I, eta, Q):
    """
    Run this function in a loop, update the state of charge
    """
    z[i+1] = z[i] - 100 * I[i]*eta*(t[i+1]-t[i]) / (Q/1000*3600)  # *100 to convert to %

