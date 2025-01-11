# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Create a simple 2D dataset with two classes
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_clusters_per_class=1, random_state=42)

# Train a Support Vector Machine Classifier
svm_model = SVC(kernel='linear')  # Use a linear kernel
svm_model.fit(X, y)

# Plot the points (data)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', marker='o', edgecolor='k', s=100)

# Plot the hyperplane (decision boundary)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create a grid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Highlight support vectors
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=200, facecolors='none', edgecolors='black', linewidth=2, label="Support Vectors")


# Add labels and title
plt.title("SVM Hyperplane and Support Vectors (2D Example)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
