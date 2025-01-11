import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Define the points
X = np.array([[-3, 0], [3, 3]])  # Points p1(-3, 0) and p2(3, 3)
y = np.array([0, 1])  # Class labels for p1 and p2

# Train an SVM classifier with a linear kernel
clf = SVC(kernel='linear')
clf.fit(X, y)

# Get the hyperplane parameters (w and b)
w = clf.coef_[0]  # weight vector
b = clf.intercept_[0]  # bias term

# Define the decision boundary equation: w.x + b = 0
# We'll use this to calculate the margin
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the points, hyperplane, and margins
plt.figure(figsize=(8, 6))

# Plot the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='o', s=100, edgecolors='k')

# Plot the hyperplane (decision boundary)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Plot the margins
plt.contour(xx, yy, Z, levels=[1], linewidths=1, colors='red', linestyles='--')
plt.contour(xx, yy, Z, levels=[-1], linewidths=1, colors='red', linestyles='--')

# Plot support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, facecolors='none', edgecolors='k', linewidths=2)


# Label the axes and show plot
plt.title('SVM Linear Kernel: Hyperplane and Margins')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.show()

# Output hyperplane equation
print(f"Hyperplane equation: w1*x + w2*y + b = 0")
print(f"w = {w}")
print(f"b = {b}")
