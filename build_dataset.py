# build_dataset.py

import os
import numpy as np
from tqdm import tqdm

# Import your custom processing modules
from preprocess import preprocess_clipped_signal, apply_smoothing
from time_domain_features import extract_time_domain_features
from fft_features import extract_frequency_domain_features
from mwt_features import extract_mwt_features
from dwt_features import extract_dwt_features
from wpt_features import extract_wpt_features

# Configuration
DATA_DIRECTORY = '/home/adisharmaruda/ee656/AirCompressor_Data/AirCompressor_Data'
OUTPUT_FEATURES_FILE = 'all_features.npy'
OUTPUT_LABELS_FILE = 'all_labels.npy'

def extract_all_features(preprocessed_signal):
    """
    A helper function to run the full 286-feature extraction pipeline.
    """
    td_f = extract_time_domain_features(preprocessed_signal)
    fft_f = extract_frequency_domain_features(preprocessed_signal)
    mwt_f = extract_mwt_features(preprocessed_signal)
    dwt_f = extract_dwt_features(preprocessed_signal, smoothing_func=apply_smoothing)
    wpt_f = extract_wpt_features(preprocessed_signal)
    
    final_feature_vector = np.concatenate([td_f, fft_f, mwt_f, dwt_f, wpt_f])
    return final_feature_vector

def process_all_data():
    """
    Iterates through the data directory, processes each file, and saves the
    final feature and label arrays.
    """
    print(f"--- Starting Data Processing from Directory: {DATA_DIRECTORY} ---")

    # Define the class labels
    class_folders = sorted(os.listdir(DATA_DIRECTORY))
    label_map = {folder_name: i for i, folder_name in enumerate(class_folders)}
    
    print("\nAssigned Class Labels:")
    for name, label in label_map.items():
        print(f"- {name}: {label}")

    master_feature_list = []
    master_label_list = []

    # Iterate through each class folder
    for folder_name, label in label_map.items():
        folder_path = os.path.join(DATA_DIRECTORY, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Get all .dat files in the folder
        files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
        
        print(f"\nProcessing folder: '{folder_name}' ({len(files_in_folder)} files)")
        
        for filename in tqdm(files_in_folder, desc=f"Folder {folder_name}"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Load the signal from the .dat file
                clipped_signal = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
                
                if clipped_signal.size != 50000:
                    print(f"Warning: Skipping {filename}, incorrect size: {clipped_signal.size}")
                    continue

                # Pre-process the signal
                preprocessed_signal, _ = preprocess_clipped_signal(clipped_signal)

                # Extract all 286 features
                feature_vector = extract_all_features(preprocessed_signal)
                
                # Append the results to our master lists
                master_feature_list.append(feature_vector)
                master_label_list.append(label)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Convert lists to NumPy arrays
    print("\nConverting collected data to NumPy arrays...")
    X_full = np.array(master_feature_list)
    y_full = np.array(master_label_list)
    
    print(f"Final dataset shapes: X={X_full.shape}, y={y_full.shape}")

    # Save the final arrays to disk
    print(f"Saving feature matrix to '{OUTPUT_FEATURES_FILE}'...")
    np.save(OUTPUT_FEATURES_FILE, X_full)
    
    print(f"Saving label vector to '{OUTPUT_LABELS_FILE}'...")
    np.save(OUTPUT_LABELS_FILE, y_full)
    
    print("\n--- All data has been processed and saved successfully! ---")

if __name__ == '__main__':
    process_all_data()