import numpy as np
import pywt

def extract_dwt_features(signal, smoothing_func, wavelet='db4', level=6):
    if signal.size == 0:
        return [0.0] * 9

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    d_coeffs = coeffs[1:]
    features = []

    for i in range(-1, -4, -1):
        features.append(np.var(d_coeffs[i]))

    for i in range(-4, -7, -1):
        ac = np.correlate(d_coeffs[i], d_coeffs[i], mode='full')
        features.append(np.var(ac))

    for i in range(-1, -4, -1):
        sm = smoothing_func(d_coeffs[i])
        features.append(np.mean(sm))

    return features
