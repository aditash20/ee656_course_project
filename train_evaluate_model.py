import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SELECTED_FEATURES_FILE = 'selected_features_25.npy'
LABELS_FILE = 'all_labels.npy'
FINAL_MODEL_FILE = 'svm_fault_diagnosis_model.pkl'

def train_and_evaluate():
    if not os.path.exists(SELECTED_FEATURES_FILE) or not os.path.exists(LABELS_FILE):
        return

    X = np.load(SELECTED_FEATURES_FILE)
    y = np.load(LABELS_FILE)
    class_names = sorted(os.listdir('/home/adisharmaruda/ee656/AirCompressor_Data/AirCompressor_Data'))

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', decision_function_shape='ovo'))
    ])

    params = {
        'svc__C': [0.1, 1, 10, 100, 1000],
        'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = GridSearchCV(pipe, params, cv=cv, scoring='accuracy', verbose=0, n_jobs=-1)
    clf.fit(X, y)

    model = clf.best_estimator_
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print(classification_report(y, y_pred, target_names=class_names))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    joblib.dump(model, FINAL_MODEL_FILE)

if __name__ == '__main__':
    train_and_evaluate()
