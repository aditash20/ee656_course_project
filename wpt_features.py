# features/wpt_features.py

import numpy as np
import pywt

def extract_wpt_features(signal, wavelet='db4', level=7):
    """
    Extracts 254 features from the Wavelet Packet Transform.
    The feature is the energy of each node in the tree (excluding the root).
    Corresponds to Section V-C-3 of the paper.

    Args:
        signal (np.ndarray): The pre-processed input signal.
        wavelet (str): The name of the wavelet to use (paper uses 'db4').
        level (int): The level of decomposition (paper uses 7).

    Returns:
        list: A list containing the 254 WPT-based feature values.
    """
    if signal.size == 0:
        return [0.0] * 254

    # Create the Wavelet Packet tree.
    wpt = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
    
    features = []
    # Iterate through all levels of the tree, from 1 to the max level.
    for lvl in range(1, level + 1):
        # Get all nodes at the current level in 'natural' (frequency) order.
        nodes = wpt.get_level(lvl, order='natural')
        for node in nodes:
            # The feature is the energy (sum of squared coefficients) of this node.
            energy = np.sum(node.data**2)
            features.append(energy)
            
    # The total number of features should be: 2^1 + 2^2 + ... + 2^7 = 254.
    return features

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("--- Running Standalone Test for wpt_features.py ---")
    sample_signal = np.random.randn(50000)
    
    wpt_features = extract_wpt_features(sample_signal)
    
    print(f"Successfully extracted {len(wpt_features)} features.")
    print("\n--- Test Results (first 5 and last 5 features) ---")
    print("First 5:", np.round(wpt_features[:5], 4))
    print("Last 5:", np.round(wpt_features[-5:], 4))