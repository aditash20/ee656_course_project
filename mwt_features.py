import numpy as np
import pywt
from scipy import stats
from scipy.signal import find_peaks

def _calculate_entropy(c):
    if c.size == 0:
        return 0.0
    a = np.abs(c)
    s = np.sum(a)
    if s == 0:
        return 0.0
    p = a / s
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def extract_mwt_features(x, a=16, b=0):
    if x.size == 0:
        return [0.0] * 7

    c, _ = pywt.cwt(x, [a], 'morl')
    c = c.flatten()

    e = _calculate_entropy(c)
    a_c = np.abs(c)
    p, _ = find_peaks(a_c)
    s_p = np.sum(a_c[p]) if len(p) > 0 else 0.0
    sd = np.std(c)
    k = stats.kurtosis(c)
    zcr = np.sum(np.diff(np.sign(np.real(c))) != 0) / len(c)
    v = np.var(c)
    s = stats.skew(c)

    return [e, s_p, sd, k, zcr, v, s]
