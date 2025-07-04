import numpy as np
import matplotlib.pyplot as plt

def apply_smoothing(signal, q=2):
    window_size = 2 * q + 1
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

def apply_robust_normalization(signal, a=-1, b=1, reject_percentile=0.025):
    X_min = np.percentile(signal, reject_percentile)
    X_max = np.percentile(signal, 100 - reject_percentile)
    if (X_max - X_min) == 0:
        return np.full_like(signal, (a + b) / 2)
    return a + ((signal - X_min) * (b - a)) / (X_max - X_min)

def preprocess_clipped_signal(clipped_signal):
    smoothed = apply_smoothing(clipped_signal)
    final = apply_robust_normalization(smoothed)
    return final, {'clipped': clipped_signal, 'smoothed': smoothed, 'final': final}

def plot_processing_steps(steps):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(steps['clipped'])
    axes[0].set_title("Clipped Signal")
    axes[1].plot(steps['smoothed'])
    axes[1].set_title("Smoothed Signal")
    axes[2].plot(steps['final'])
    axes[2].set_title("Final Signal")
    axes[2].axhline(y=1.0, color='r', linestyle='--', lw=1)
    axes[2].axhline(y=-1.0, color='r', linestyle='--', lw=1)
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    signal = np.loadtxt('/home/adisharmaruda/ee656/AirCompressor_Data/Compressor_Fault_Diagnosis/Data/Bearing/preprocess_Reading1.txt',
                        dtype=np.float32, delimiter=',')
    result, steps = preprocess_clipped_signal(signal)
    plot_processing_steps(steps)
