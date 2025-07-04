# features/mwt_features.py

import numpy as np
import pywt
from scipy import stats
from scipy.signal import find_peaks

def _calculate_entropy(coeffs):
    """
    Helper function to calculate Shannon Entropy following equation (18) from the paper.
    Uses normalized coefficient magnitudes as probabilities.
    """
    if coeffs.size == 0:
        return 0.0
    
    abs_coeffs = np.abs(coeffs)
    total_energy = np.sum(abs_coeffs)
    
    if total_energy == 0:
        return 0.0
    
    probabilities = abs_coeffs / total_energy
    
    probabilities = probabilities[probabilities > 0]
    
    return -np.sum(probabilities * np.log2(probabilities))

def extract_mwt_features(signal, scale_param=16, translation_param=0):
    """
    Extracts 7 features from the Morlet Wavelet Transform of the signal.
    Corresponds to Section V-C-1 of the paper.
    
    The paper specifies using Morlet wavelet with scaling parameter 'a' and 
    translation parameter 'b'. Default values match the paper's specifications.
    
    Args:
        signal (np.ndarray): The pre-processed input signal.
        scale_param (float): Scaling parameter 'a' (paper uses 16).
        translation_param (float): Translation parameter 'b' (paper uses 0).

    Returns:
        list: A list containing the 7 MWT-based feature values.
    """
    if signal.size == 0:
        return [0.0] * 7

    # Apply Continuous Wavelet Transform using Morlet wavelet
    scales = [scale_param]
    cwt_coeffs, _ = pywt.cwt(signal, scales, 'morl')
    
    coeffs = cwt_coeffs.flatten()

    # Feature 1: Wavelet Entropy
    entropy = _calculate_entropy(coeffs)
    
    # Feature 2: Sum of Peaks
    abs_coeffs = np.abs(coeffs)
    peaks, _ = find_peaks(abs_coeffs)
    sum_of_peaks = np.sum(abs_coeffs[peaks]) if len(peaks) > 0 else 0.0
    
    # Feature 3: Standard Deviation of coefficients
    std_dev = np.std(coeffs)
    
    # Feature 4: Kurtosis of coefficients
    kurtosis_mwt = stats.kurtosis(coeffs)
    
    # Feature 5: Zero Crossing Rate
    real_coeffs = np.real(coeffs)
    zero_crossings = np.sum(np.diff(np.sign(real_coeffs)) != 0) / len(coeffs)
    
    # Feature 6: Variance of coefficients
    variance_mwt = np.var(coeffs)
    
    # Feature 7: Skewness of coefficients
    skewness_mwt = stats.skew(coeffs)
    
    feature_vector = [
        entropy, sum_of_peaks, std_dev, kurtosis_mwt, 
        zero_crossings, variance_mwt, skewness_mwt
    ]
    
    return feature_vector

