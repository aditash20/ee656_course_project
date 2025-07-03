from preprocess import preprocess_clipped_signal, plot_processing_steps
from time_domain_features import extract_time_domain_features
from fft_features import extract_frequency_domain_features
import numpy as np
input_data = np.loadtxt('/home/adisharmaruda/ee656/AirCompressor_Data/Compressor_Fault_Diagnosis/Data/Bearing/preprocess_Reading2.txt', 
                                dtype=np.float32, delimiter=',')
print(input_data.shape,"djksj")
# 2. Run the pre-processing pipeline for clipped data
preprocessed_signal, all_steps = preprocess_clipped_signal(input_data)

# 3. Visualize the results to verify the process
# plot_processing_steps(all_steps)

# 4. Extract time domain features
time_domain_features = extract_time_domain_features(preprocessed_signal)
print(len(time_domain_features), "time_domain_features")

# 5. Extract frequency domain features
frequency_domain_features = extract_frequency_domain_features(preprocessed_signal)
print(len(frequency_domain_features), "frequency_domain_features")

# 6. Combine all features
all_features = time_domain_features + frequency_domain_features
print(len(all_features), "all_features")
print(f"\nSum of all feature ratios: {np.sum(frequency_domain_features):.4f}")

