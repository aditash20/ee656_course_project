# features/dwt_features.py

import numpy as np
import pywt

def extract_dwt_features(signal, smoothing_func, wavelet='db4', level=6):
    """
    Extracts 9 features from the Discrete Wavelet Transform.
    Corresponds to Section V-C-2 of the paper.

    Args:
        signal (np.ndarray): The pre-processed input signal.
        smoothing_func (function): The moving average smoothing function from the pre-processing step.
        wavelet (str): The name of the wavelet to use (paper uses 'db4').
        level (int): The level of decomposition (paper uses 6).

    Returns:
        list: A list containing the 9 DWT-based feature values.
    """
    if signal.size == 0:
        return [0.0] * 9

    # Decompose the signal into wavelet coefficients.
    # Coeffs are ordered [cA6, cD6, cD5, cD4, cD3, cD2, cD1].
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # We only need the detail coefficients (D1 to D6).
    # Indexing from the end is convenient: d_coeffs[-1] is D1.
    d_coeffs = coeffs[1:]
    
    features = []
    
    # Features 1-3: Variance of detail coefficients at levels 1, 2, 3.
    for i in range(-1, -4, -1):  # Corresponds to D1, D2, D3
        features.append(np.var(d_coeffs[i]))
        
    # Features 4-6: Variance of the autocorrelation of detail coefficients at levels 4, 5, 6.
    for i in range(-4, -7, -1):  # Corresponds to D4, D5, D6
        autocorr = np.correlate(d_coeffs[i], d_coeffs[i], mode='full')
        features.append(np.var(autocorr))
        
    # Features 7-9: Statistical mean of the smoothed detail coefficients at levels 1, 2, 3.
    for i in range(-1, -4, -1):  # Corresponds to D1, D2, D3
        smoothed_d = smoothing_func(d_coeffs[i])
        features.append(np.mean(smoothed_d))
        
    return features

# --- Standalone Test Block ---
if __name__ == '__main__':
    # For standalone testing, we need to define a dummy smoothing function.
    def dummy_smoothing(sig, q=2):
        return np.convolve(sig, np.ones(2*q+1)/(2*q+1), mode='same')

    print("--- Running Standalone Test for dwt_features.py ---")
    sample_signal = np.random.randn(50000)
    
    dwt_features = extract_dwt_features(sample_signal, smoothing_func=dummy_smoothing)
    
    print(f"Successfully extracted {len(dwt_features)} features.")
    print("\n--- Test Results (first 3 features) ---")
    print(np.round(dwt_features[:3], 4))