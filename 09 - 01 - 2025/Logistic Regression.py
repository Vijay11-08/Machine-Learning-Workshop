import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features for 2D visualization
y = iris.target

# Binarize the target labels for binary classification
# This will transform the problem into a one-vs-rest binary classification for each class
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)  # One-hot encode the target labels

# Train Logistic Regression classifier (one-vs-rest strategy)
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X, y)


# Visualize the decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Logistic Regression Decision Boundary - Iris Dataset")
    plt.xlabel("Feature 1 (Sepal Length)")
    plt.ylabel("Feature 2 (Sepal Width)")
    plt.show()

plot_decision_boundary(X, y, model)
