# features/fft_features.py

import numpy as np
from scipy.fft import fft

def extract_frequency_domain_features(signal, n_bins=8):
    """
    Extracts 8 features based on the energy ratio in positive frequency bins.
    This is the standard and most robust method for real-valued signals.

    Args:
        signal (np.ndarray): The pre-processed input signal.
        n_bins (int): The number of frequency bins to create (paper uses 8).

    Returns:
        list: A list containing the 8 calculated feature values.
    """
    if signal.size == 0:
        return [0.0] * n_bins

    N = len(signal)
    
    # 1. Compute the FFT of the signal.
    fft_vals = fft(signal)
    
    # 2. Compute the Power Spectral Density (PSD) using only positive frequencies.
    #    For real signals, all information is in the first half of the spectrum.
    psd = (2.0 / N) * np.abs(fft_vals[0:N//2])**2
    
    # 3. Calculate the total energy in the positive frequency spectrum.
    total_energy = np.sum(psd)
    
    if total_energy == 0:
        return [0.0] * n_bins
        
    # 4. Divide the positive spectrum into n_bins and calculate the energy ratio.
    points_per_bin = len(psd) // n_bins
    
    features = []
    for i in range(n_bins):
        start_idx = i * points_per_bin
        end_idx = (i + 1) * points_per_bin
        
        # Ensure the last bin captures all remaining points.
        if i == n_bins - 1:
            end_idx = len(psd)
            
        bin_energy = np.sum(psd[start_idx:end_idx])
        energy_ratio = bin_energy / total_energy
        features.append(energy_ratio)
        
    return features

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("--- Running Standalone Test for fft_features.py ---")
    
    # Generate some sample data
    num_points = 50000
    time = np.linspace(0, 1, num_points)
    sample_signal = (1.0 * np.sin(2 * np.pi * 1000 * time) + 
                     0.5 * np.sin(2 * np.pi * 8000 * time) +
                     0.1 * np.random.randn(num_points))
                     
    # Extract features using the recommended implementation
    fft_features = extract_frequency_domain_features(sample_signal)
    
    print(f"Successfully extracted {len(fft_features)} features.")
    
    print("\n--- Test Results ---")
    for i, value in enumerate(fft_features):
        print(f"Energy Ratio in Bin {i+1}: {value:.4f}")
        
    print(f"\nSum of all feature ratios: {np.sum(fft_features):.4f}")