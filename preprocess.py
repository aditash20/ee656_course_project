import numpy as np
import matplotlib.pyplot as plt

# --- Main Pre-processing Functions for Clipped Data ---

def apply_smoothing(signal, q=2):
    """
    Applies a moving average filter to the signal as per Section IV-C.
    For q=2, the window size is 2*q + 1 = 5.

    Args:
        signal (np.ndarray): The input signal (assumed to be 50,000 points).
        q (int): The number of samples to consider on each side (paper uses q=2).

    Returns:
        np.ndarray: The smoothed signal.
    """
    print("Step 1: Applying smoothing filter...")
    window_size = 2 * q + 1
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, kernel, mode='same')
    print(f"-> Smoothing complete with window size {window_size}.")
    return smoothed_signal

def apply_robust_normalization(signal, a=-1, b=1, reject_percentile=0.025):
    """
    Normalizes the signal using the robust method from Section IV-D.
    This method finds min/max values after excluding a small percentage
    of the data from both ends of the distribution to avoid distortion from outliers.

    Args:
        signal (np.ndarray): The input signal to normalize.
        a (float): The lower bound of the target scale (default -1).
        b (float): The upper bound of the target scale (default 1).
        reject_percentile (float): The percentage of data to ignore at each tail.

    Returns:
        np.ndarray: The normalized signal.
    """
    print("Step 2: Applying robust normalization...")
    
    # Use percentiles for a direct and robust way to find X_min and X_max
    X_min = np.percentile(signal, reject_percentile)
    X_max = np.percentile(signal, 100 - reject_percentile)

    print(f"-> Original Min/Max: {signal.min():.4f} / {signal.max():.4f}")
    print(f"-> Robust Min/Max for scaling (based on {reject_percentile}% tails): {X_min:.4f} / {X_max:.4f}")

    if (X_max - X_min) == 0:
        return np.full_like(signal, (a + b) / 2) # Avoid division by zero
        
    normalized_signal = a + ((signal - X_min) * (b - a)) / (X_max - X_min)
    print("-> Normalization complete.")
    return normalized_signal

# --- Main Pipeline Orchestrator for Clipped Data ---

def preprocess_clipped_signal(clipped_signal):
    """
    Runs the pre-processing pipeline on an already clipped signal.

    Args:
        clipped_signal (np.ndarray): The signal from a .dat file (50,000 points).

    Returns:
        np.ndarray: The fully pre-processed signal, ready for feature extraction.
        dict: A dictionary containing the signal at each intermediate step for plotting.
    """
    print("\n--- Starting Pre-processing Pipeline for Clipped Data ---")
    
    # Step 1: Smoothing
    smoothed_signal = apply_smoothing(clipped_signal)
    
    # Step 2: Normalization
    final_signal = apply_robust_normalization(smoothed_signal)
    
    print("--- Pre-processing Complete ---")
    
    # Store intermediate results for visualization
    processing_steps = {
        'clipped': clipped_signal,
        'smoothed': smoothed_signal,
        'final': final_signal
    }
    
    return final_signal, processing_steps

# --- Helper Functions for Demonstration ---

def generate_synthetic_clipped_signal(num_points=50000):
    """Generates a synthetic 1-second clipped signal for testing."""
    print("Generating a synthetic 1-second (50k points) clipped signal...")
    time = np.linspace(0, 1, num_points)
    
    # A base noisy signal
    signal = 0.5 * np.sin(2 * np.pi * 120 * time) + 0.2 * np.random.randn(num_points)
    
    # Add extreme outliers
    signal[10000] = 8.0
    signal[40000] = -7.0
    
    return signal

def plot_processing_steps(steps):
    """Visualizes the signal at each stage of pre-processing for clipped data."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle("Pre-processing Stages for Clipped Signal", fontsize=16)

    # Plot 1: Clipped Signal (Input)
    axes[0].plot(steps['clipped'])
    axes[0].set_title(f"1. Input Clipped Signal ({len(steps['clipped'])} points)")
    axes[0].set_ylabel("Amplitude")

    # Plot 2: Smoothed Signal
    axes[1].plot(steps['smoothed'])
    axes[1].set_title(f"2. After Smoothing ({len(steps['smoothed'])} points)")
    axes[1].set_ylabel("Amplitude")
    
    # Plot 3: Final Normalized Signal
    axes[2].plot(steps['final'])
    axes[2].set_title(f"3. Final Pre-processed Signal ({len(steps['final'])} points)")
    axes[2].set_xlabel("Sample Number")
    axes[2].set_ylabel("Normalized Amp.")
    axes[2].axhline(y=1.0, color='r', linestyle='--', lw=1)
    axes[2].axhline(y=-1.0, color='r', linestyle='--', lw=1)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # --- Main Execution Block ---
    
    # 1. Load your clipped .dat file here
    # For this example, we generate a synthetic signal.
    # Replace this line with your actual data loading code:
    # clipped_signal = np.fromfile('path/to/your_file.dat', dtype=np.float32)
    # clipped_signal = generate_synthetic_clipped_signal()
    clipped_signal = np.loadtxt('/home/adisharmaruda/ee656/AirCompressor_Data/Compressor_Fault_Diagnosis/Data/Bearing/preprocess_Reading1.txt', 
                                dtype=np.float32, delimiter=',')
    print(clipped_signal.shape,"djksj")
    # 2. Run the pre-processing pipeline for clipped data
    preprocessed_signal, all_steps = preprocess_clipped_signal(clipped_signal)
    
    # 3. Visualize the results to verify the process
    plot_processing_steps(all_steps)
    
    # The 'preprocessed_signal' is now ready for the feature extraction stage.
    print(f"\nFinal pre-processed signal shape: {preprocessed_signal.shape}")