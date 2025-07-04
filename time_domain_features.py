import numpy as np
from scipy import stats

def extract_time_domain_features(signal):
    if signal.size == 0:
        return [0.0] * 8

    a = np.abs(signal)
    m = np.mean(a)
    p = np.max(a)
    rms = np.sqrt(np.mean(signal ** 2))
    v = np.var(signal)
    k = stats.kurtosis(signal)
    s = stats.skew(signal)
    c = p / rms if rms != 0 else 0
    sf = rms / m if m != 0 else 0

    return [m, p, rms, v, k, s, c, sf]
