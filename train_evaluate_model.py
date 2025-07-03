# train_evaluate_model.py

import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Configuration ---
# Input files from the feature selection step
SELECTED_FEATURES_FILE = 'selected_features_25.npy'
LABELS_FILE = 'all_labels.npy'

# Output file for the final trained model
FINAL_MODEL_FILE = 'svm_fault_diagnosis_model.pkl'

# --- Main Script Logic ---

def train_and_evaluate():
    """
    Loads the selected features, trains an SVM using GridSearchCV with
    cross-validation, evaluates its performance, and saves the best model.
    """
    print("--- SVM Model Training and Evaluation ---")

    # 1. Load the final, optimized dataset
    if not os.path.exists(SELECTED_FEATURES_FILE) or not os.path.exists(LABELS_FILE):
        print(f"Error: Input files not found!")
        print(f"Please create '{SELECTED_FEATURES_FILE}' and '{LABELS_FILE}' first.")
        return

    print(f"Loading data from '{SELECTED_FEATURES_FILE}' and '{LABELS_FILE}'...")
    X = np.load(SELECTED_FEATURES_FILE)
    y = np.load(LABELS_FILE)
    print(f"Data loaded. Shapes: X={X.shape}, y={y.shape}")
    
    # Define class names for reporting (make sure this order matches your label_map)
    class_names = sorted(os.listdir('/home/adisharmaruda/ee656/AirCompressor_Data/AirCompressor_Data'))

    # 2. Set up the training pipeline and parameter grid for Grid Search
    # A Pipeline is a great way to chain steps. Here, we scale the data then classify.
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Step 1: Scale features for SVM
        ('svc', SVC(kernel='rbf', decision_function_shape='ovo')) # Step 2: SVM classifier
    ])
    # 'ovo' (One-vs-One) is what the paper recommends and is the scikit-learn default.

    # Define the hyperparameter grid to search, as per the paper's method.
    # The paper mentions log2(C) and log2(gamma), so we use powers of 10 or 2.
    param_grid = {
        'svc__C': [0.1, 1, 10, 100, 1000],          # Regularization parameter
        'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001] # Kernel coefficient
    }

    # 3. Set up k-fold cross-validation
    # The paper uses k=5 and k=10. Let's use 5 as a robust choice.
    # StratifiedKFold ensures each fold has a proportional representation of each class.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Perform Grid Search with Cross-Validation
    # This will automatically train and evaluate the SVM for every combination
    # of parameters across all 5 folds.
    print("\nStarting Grid Search with 5-fold Cross-Validation...")
    print("This may take some time...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='accuracy', # The metric to optimize
        verbose=1,         # Shows progress
        n_jobs=-1          # Use all available CPU cores
    )
    
    # Fit the grid search to the entire dataset. It handles the splitting internally.
    grid_search.fit(X, y)

    # 5. Report the results
    print("\n--- Grid Search Results ---")
    print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
    print(f"Best parameters found: {grid_search.best_params_}")

    # The grid_search automatically retrains the best model on the ENTIRE dataset
    best_model = grid_search.best_estimator_
    
    # To get a confusion matrix, we need predictions. We can get them via cross_val_predict.
    from sklearn.model_selection import cross_val_predict
    print("\nGenerating confusion matrix based on cross-validated predictions...")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
    
    # 6. Final Evaluation Metrics
    print("\n--- Final Model Performance (from Cross-Validation) ---")
    print(classification_report(y, y_pred_cv, target_names=class_names))

    # Plot the confusion matrix
    cm = confusion_matrix(y, y_pred_cv)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Cross-Validated Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # 7. Save the final, trained model
    print(f"\nSaving the best trained model to '{FINAL_MODEL_FILE}'...")
    joblib.dump(best_model, FINAL_MODEL_FILE)
    print("-> Model saved successfully!")
    print("\nThis model can now be used for predicting new, unseen data.")

if __name__ == '__main__':
    train_and_evaluate()