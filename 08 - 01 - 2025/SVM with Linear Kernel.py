import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Generate a sample dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)

# Create and fit the SVM model
model = SVC(kernel='linear', C=1)
model.fit(X, y)

# Plot the data points
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k', label="Data Points")

# Get the separating hyperplane
w = model.coef_[0]
b = model.intercept_[0]
x_hyperplane = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_hyperplane = -(w[0] * x_hyperplane + b) / w[1]
plt.plot(x_hyperplane, y_hyperplane, color='black', label="Hyperplane")

# Plot the margin lines
margin = 1 / np.sqrt(np.sum(w**2))
y_margin_upper = y_hyperplane + margin
y_margin_lower = y_hyperplane - margin
plt.plot(x_hyperplane, y_margin_upper, 'r--', label="Margin")
plt.plot(x_hyperplane, y_margin_lower, 'r--')

# Highlight support vectors
support_vectors = model.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200, facecolors='none', edgecolors='k', label="Support Vectors")

# Plot customization
plt.title("SVM with Linear Kernel", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
