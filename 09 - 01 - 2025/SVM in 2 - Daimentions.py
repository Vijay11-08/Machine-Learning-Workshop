import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Generate a dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')  # Linear SVM
clf.fit(X, y)

# Plotting the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o')
plt.title('SVM in 2D (Linear)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')


# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

plt.show()
