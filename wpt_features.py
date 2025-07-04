import numpy as np
import pywt

def extract_wpt_features(signal, wavelet='db4', level=7):
    if signal.size == 0:
        return [0.0] * 254

    wpt = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    features = []

    for lvl in range(1, level + 1):
        nodes = wpt.get_level(lvl, order='natural')
        for node in nodes:
            energy = np.sum(node.data ** 2)
            features.append(energy)

    return features
