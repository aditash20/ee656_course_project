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
    
    # Use absolute values of coefficients and normalize to get probabilities
    abs_coeffs = np.abs(coeffs)
    total_energy = np.sum(abs_coeffs)
    
    if total_energy == 0:
        return 0.0
    
    # Calculate probabilities as normalized absolute coefficients
    probabilities = abs_coeffs / total_energy
    
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate Shannon entropy: H = -sum(p_i * log2(p_i))
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
    # The scale parameter corresponds to 'a' in the paper's equation (17)
    scales = [scale_param]
    cwt_coeffs, _ = pywt.cwt(signal, scales, 'morl')
    
    # The result is a 2D array (1 row for each scale), so we flatten it
    coeffs = cwt_coeffs.flatten()

    # Feature 1: Wavelet Entropy (following equation 18)
    entropy = _calculate_entropy(coeffs)
    
    # Feature 2: Sum of Peaks (sum of peak magnitudes, not just count)
    abs_coeffs = np.abs(coeffs)
    peaks, _ = find_peaks(abs_coeffs)
    sum_of_peaks = np.sum(abs_coeffs[peaks]) if len(peaks) > 0 else 0.0
    
    # Feature 3: Standard Deviation of coefficients
    std_dev = np.std(coeffs)
    
    # Feature 4: Kurtosis of coefficients
    kurtosis_mwt = stats.kurtosis(coeffs)
    
    # Feature 5: Zero Crossing Rate (normalized by length)
    # For complex coefficients, we examine the real part
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

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("--- Running Standalone Test for mwt_features.py ---")
    
    # Test with different signal types
    print("\n=== Test 1: Random Signal ===")
    sample_signal = np.random.randn(50000)
    mwt_features = extract_mwt_features(sample_signal)
    
    print(f"Successfully extracted {len(mwt_features)} features.")
    feature_names = ["Entropy", "Sum of Peaks", "Std Dev", "Kurtosis", "Zero Cross", "Variance", "Skewness"]
    
    print("\n--- Feature Results ---")
    for name, value in zip(feature_names, mwt_features):
        print(f"{name:>15}: {value:.6f}")
    
    # Test with a sinusoidal signal for validation
    print("\n=== Test 2: Sinusoidal Signal ===")
    t = np.linspace(0, 1, 50000)
    sin_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(50000)
    mwt_features_sin = extract_mwt_features(sin_signal)
    
    print("--- Feature Results ---")
    for name, value in zip(feature_names, mwt_features_sin):
        print(f"{name:>15}: {value:.6f}")
    
    # Test edge case: empty signal
    print("\n=== Test 3: Edge Case (Empty Signal) ===")
    empty_signal = np.array([])
    mwt_features_empty = extract_mwt_features(empty_signal)
    print(f"Empty signal features: {mwt_features_empty}")
    
    print("\n--- All Tests Completed Successfully ---")