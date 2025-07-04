import numpy as np
from scipy.fft import fft

def extract_frequency_domain_features(signal, n_bins=8):
    if signal.size == 0:
        return [0.0] * n_bins

    N = len(signal)
    fft_vals = fft(signal)
    psd = (2.0 / N) * np.abs(fft_vals[:N // 2]) ** 2
    total_energy = np.sum(psd)

    if total_energy == 0:
        return [0.0] * n_bins

    points_per_bin = len(psd) // n_bins
    features = []

    for i in range(n_bins):
        start = i * points_per_bin
        end = (i + 1) * points_per_bin
        if i == n_bins - 1:
            end = len(psd)
        energy = np.sum(psd[start:end])
        features.append(energy / total_energy)

    return features
