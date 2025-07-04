import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import classification_report

# Configuration
X = np.load("all_features.npy")
y = np.load("all_labels.npy")
k = 25
methods = ['pca', 'mifs', 'bd']
runtime_dict = {}
accuracy_dict = {}
reports = {}
class_names = ['Bearing', 'Flywheel', 'Healthy', 'LIV', 'LOV', 'NRV', 'Piston', 'Riderbelt']

# Selection Methods
def select_pca(X, k):
    return PCA(n_components=k).fit_transform(X)

def select_mifs(X, y, k=25, beta=0.5):
    mi = mutual_info_classif(X, y)
    selected = [np.argmax(mi)]
    for _ in range(1, k):
        best_score = -np.inf
        best_feat = -1
        for j in range(X.shape[1]):
            if j in selected:
                continue
            redundancy = np.mean([
                mutual_info_regression(X[:, [j]], X[:, [s]].ravel())[0] for s in selected
            ])
            score = mi[j] - beta * redundancy
            if score > best_score:
                best_score = score
                best_feat = j
        selected.append(best_feat)
    return selected

def select_bd(X, y, k):
    scores = []
    for i in range(X.shape[1]):
        dists = []
        for cls in np.unique(y):
            x_cls = X[y == cls, i]
            mean = np.mean(x_cls)
            var = np.var(x_cls) + 1e-6
            dists.append((mean ** 2) / var)
        scores.append(np.mean(dists))
    return np.argsort(scores)[-k:]

# Evaluation Loop
for method in methods:
    print(f"\n--- {method.upper()} ---")
    start = time.time()

    if method == 'pca':
        X_sel = select_pca(X, k)
    elif method == 'mifs':
        selected_indices = select_mifs(X, y, k)
        X_sel = X[:, selected_indices]
    elif method == 'bd':
        selected_indices = select_bd(X, y, k)
        X_sel = X[:, selected_indices]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', decision_function_shape='ovo'))
    ])
    y_pred = cross_val_predict(model, X_sel, y, cv=StratifiedKFold(n_splits=5))

    end = time.time()
    runtime = end - start
    acc = np.mean(y_pred == y) * 100

    runtime_dict[method] = runtime
    accuracy_dict[method] = acc
    reports[method] = classification_report(y, y_pred, target_names=class_names, output_dict=True)

    print(f"Accuracy: {acc:.2f}% | Runtime: {runtime:.2f}s")

# Plot Bar Chart: Accuracy
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, [accuracy_dict[m] for m in methods], color='skyblue')
for bar, method in zip(bars, methods):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{accuracy_dict[method]:.2f}%", ha='center')
plt.ylabel("Accuracy (%)")
plt.title(f"Accuracy Comparison at k={k}")
plt.tight_layout()
plt.savefig("accuracy_comparison_final.png", dpi=300)
plt.show()

# Plot Bar Chart: Runtime
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, [runtime_dict[m] for m in methods], color='orange')
for bar, method in zip(bars, methods):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{runtime_dict[method]:.2f}s", ha='center')
plt.ylabel("Runtime (s)")
plt.title(f"Runtime Comparison at k={k}")
plt.tight_layout()
plt.savefig("runtime_comparison_final.png", dpi=300)
plt.show()

# Optional: Print reports
for method in methods:
    print(f"\nClassification Report for {method.upper()}:\n")
    print(pd.DataFrame(reports[method]).T.round(2))