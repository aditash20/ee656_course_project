import numpy as np
import pandas as pd
from mrmr import mrmr_classif
import os
import joblib

NUM_FEATURES_TO_SELECT = 25
FULL_FEATURES_FILE = 'all_features.npy'
FULL_LABELS_FILE = 'all_labels.npy'
SELECTED_INDICES_FILE = 'Feature_Select_mRMR_25.pkl'
REDUCED_DATASET_FILE = 'selected_features_25.npy'

def select_and_save_best_features(k):
    if not os.path.exists(FULL_FEATURES_FILE) or not os.path.exists(FULL_LABELS_FILE):
        X_dummy = np.random.rand(200, 286)
        y_dummy = np.random.randint(0, 8, 200)
        np.save(FULL_FEATURES_FILE, X_dummy)
        np.save(FULL_LABELS_FILE, y_dummy)

    X = np.load(FULL_FEATURES_FILE)
    y = np.load(FULL_LABELS_FILE)

    feature_names = [f'f_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_s = pd.Series(y)

    selected_names = mrmr_classif(X=X_df, y=y_s, K=k)
    selected_indices = [feature_names.index(name) for name in selected_names]

    joblib.dump(selected_indices, SELECTED_INDICES_FILE)
    X_selected = X[:, selected_indices]
    np.save(REDUCED_DATASET_FILE, X_selected)

if __name__ == '__main__':
    select_and_save_best_features(NUM_FEATURES_TO_SELECT)
