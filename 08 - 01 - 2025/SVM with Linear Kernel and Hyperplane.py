# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load a sample dataset (e.g., Iris dataset)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Taking the first two features for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the SVM model with a linear kernel
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)

# Create a grid to plot decision boundaries
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 500),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 500))

# Get the decision function
Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

# Plot the decision boundary and margin
plt.contour(xx, yy, Z, levels=[-1, 0, 1], cmap="coolwarm", alpha=0.75)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap="autumn", marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap="winter", marker='x')
plt.title("SVM with Linear Kernel and Hyperplane")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
