from preprocess import preprocess_clipped_signal
from time_domain_features import extract_time_domain_features
from fft_features import extract_frequency_domain_features
import numpy as np

data = np.loadtxt('/home/adisharmaruda/ee656/AirCompressor_Data/Compressor_Fault_Diagnosis/Data/Bearing/preprocess_Reading2.txt',
                  dtype=np.float32, delimiter=',')

signal, _ = preprocess_clipped_signal(data)

td = extract_time_domain_features(signal)
fd = extract_frequency_domain_features(signal)

features = td + fd
print(len(features))