import numpy as np
import pandas as pd
from mrmr import mrmr_classif
import os
import joblib

# Configuration
NUM_FEATURES_TO_SELECT = 25

# Input files
FULL_FEATURES_FILE = 'all_features.npy'
FULL_LABELS_FILE = 'all_labels.npy'

# Output files
SELECTED_INDICES_FILE = 'Feature_Select_mRMR_25.pkl'
REDUCED_DATASET_FILE = 'selected_features_25.npy'

def select_and_save_best_features(k):
    """
    Loads the full feature set, performs mRMR selection for the top 'k' features,
    and saves the selected indices and the reduced dataset.
    """
    print(f"--- mRMR Feature Selection (Top {k}) ---")

    # Check for and load the full dataset
    if not os.path.exists(FULL_FEATURES_FILE) or not os.path.exists(FULL_LABELS_FILE):
        print(f"Error: Input files not found!")
        print(f"Please create '{FULL_FEATURES_FILE}' and '{FULL_LABELS_FILE}' first.")
        print("Creating dummy files for this run...")
        num_samples, num_features = 200, 286
        dummy_X = np.random.rand(num_samples, num_features)
        dummy_y = np.random.randint(0, 8, num_samples)
        np.save(FULL_FEATURES_FILE, dummy_X)
        np.save(FULL_LABELS_FILE, dummy_y)
        print("Dummy files created. Please replace them with your real data.")

    print(f"Loading data from '{FULL_FEATURES_FILE}' and '{FULL_LABELS_FILE}'...")
    X_full = np.load(FULL_FEATURES_FILE)
    y = np.load(FULL_LABELS_FILE)
    print(f"Data loaded. Shapes: X={X_full.shape}, y={y.shape}")

    # Prepare data for the mrmr library
    feature_names = [f'f_{i}' for i in range(X_full.shape[1])]
    X_df = pd.DataFrame(X_full, columns=feature_names)
    y_s = pd.Series(y, name='target')

    # Perform mRMR feature selection
    print(f"Running mRMR to select the top {k} most important features...")
    selected_feature_names = mrmr_classif(X=X_df, y=y_s, K=k)
    print("mRMR selection complete.")
    
    # Convert selected feature names back to their original integer indices
    selected_indices = [feature_names.index(name) for name in selected_feature_names]
    
    print(f"Top {k} selected feature indices: {selected_indices}")

    # Save the list of selected indices to a file
    print(f"Saving selected indices to '{SELECTED_INDICES_FILE}'...")
    joblib.dump(selected_indices, SELECTED_INDICES_FILE)

    # Create and save the new, reduced dataset
    X_selected = X_full[:, selected_indices]
    print(f"Created reduced feature matrix with shape: {X_selected.shape}")
    print(f"Saving reduced dataset to '{REDUCED_DATASET_FILE}'...")
    np.save(REDUCED_DATASET_FILE, X_selected)
    
    print("--- Feature Selection complete! ---")


if __name__ == '__main__':
    select_and_save_best_features(k=NUM_FEATURES_TO_SELECT)