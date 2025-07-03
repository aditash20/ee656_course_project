# features/td_features.py

import numpy as np
from scipy import stats

def extract_time_domain_features(signal):
    """
    Extracts 8 statistical features from the time-domain signal.
    This version is designed to perfectly match Section V-A of the paper.

    Args:
        signal (np.ndarray): The pre-processed input signal.

    Returns:
        list: A list containing the 8 calculated feature values.
    """
    if signal.size == 0:
        # Handle empty signal to avoid errors
        return [0.0] * 8

    # Pre-calculate common values to improve efficiency
    abs_signal = np.abs(signal)
    
    # 1. Absolute Statistical Mean
    abs_mean = np.mean(abs_signal)
    
    # 2. Maximum Peak (of the absolute signal)
    peak = np.max(abs_signal)
    
    # 3. RMS (Root Mean Square)
    rms = np.sqrt(np.mean(np.square(signal)))
    
    # 4. Variance
    variance = np.var(signal)
    
    # 5. Kurtosis
    kurt = stats.kurtosis(signal)
    
    # 6. Skewness
    skew = stats.skew(signal)
    
    # 7. Crest Factor (handle potential division by zero)
    crest_factor = peak / rms if rms != 0 else 0
    
    # 8. Shape Factor (handle potential division by zero)
    shape_factor = rms / abs_mean if abs_mean != 0 else 0
    
    feature_vector = [
        abs_mean, 
        peak, 
        rms, 
        variance, 
        kurt, 
        skew, 
        crest_factor, 
        shape_factor
    ]
    
    return feature_vector

# --- Standalone Test Block ---
# This code only runs if you execute this file directly (e.g., "python td_features.py")
# It's useful for testing this specific module in isolation.
if __name__ == '__main__':
    print("--- Running Standalone Test for td_features.py ---")
    
    # Generate some sample data to test the function
    sample_signal = np.random.randn(50000) * 2
    sample_signal[1000] = 10  # Add a distinct peak
    
    # Call the function to extract features
    td_features = extract_time_domain_features(sample_signal)
    
    print(f"Successfully extracted {len(td_features)} features.")
    
    # Define feature names for clear output
    feature_names = [
        "Abs Mean", "Max Peak", "RMS", "Variance", 
        "Kurtosis", "Skewness", "Crest Factor", "Shape Factor"
    ]
    
    # Print the results in a readable format
    print("\n--- Test Results ---")
    for name, value in zip(feature_names, td_features):
        print(f"{name:>15}: {value:.4f}")