import os
import numpy as np
from tqdm import tqdm

from preprocess import preprocess_clipped_signal, apply_smoothing
from time_domain_features import extract_time_domain_features
from fft_features import extract_frequency_domain_features
from mwt_features import extract_mwt_features
from dwt_features import extract_dwt_features
from wpt_features import extract_wpt_features

DATA_DIRECTORY = '/home/adisharmaruda/ee656/AirCompressor_Data/AirCompressor_Data'
OUTPUT_FEATURES_FILE = 'all_features.npy'
OUTPUT_LABELS_FILE = 'all_labels.npy'

def extract_all_features(signal):
    td = extract_time_domain_features(signal)
    fft = extract_frequency_domain_features(signal)
    mwt = extract_mwt_features(signal)
    dwt = extract_dwt_features(signal, smoothing_func=apply_smoothing)
    wpt = extract_wpt_features(signal)
    return np.concatenate([td, fft, mwt, dwt, wpt])

def process_all_data():
    folders = sorted(os.listdir(DATA_DIRECTORY))
    label_map = {name: idx for idx, name in enumerate(folders)}

    features = []
    labels = []

    for name, label in label_map.items():
        folder = os.path.join(DATA_DIRECTORY, name)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith('.dat')]
        for file in tqdm(files, desc=name):
            path = os.path.join(folder, file)
            try:
                signal = np.loadtxt(path, dtype=np.float32, delimiter=',')
                if signal.size != 50000:
                    continue
                processed, _ = preprocess_clipped_signal(signal)
                vec = extract_all_features(processed)
                features.append(vec)
                labels.append(label)
            except:
                continue

    X = np.array(features)
    y = np.array(labels)

    np.save(OUTPUT_FEATURES_FILE, X)
    np.save(OUTPUT_LABELS_FILE, y)

if __name__ == '__main__':
    process_all_data()
