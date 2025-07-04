import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data
X = np.load("all_features.npy")         # shape: [n_samples, n_features]
y = np.load("all_labels.npy")           # shape: [n_samples]

# Reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", s=20)

plt.title("PCA Projection of Extracted Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("pca_feature_scatter.png", dpi=300)
plt.show()