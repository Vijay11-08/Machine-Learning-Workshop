import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate Data
# Independent variables
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([2, 4, 6, 8, 10])
X = np.column_stack((X1, X2))  # Combine X1 and X2

# Dependent variable
Y = np.array([5, 7, 9, 11, 13])

# Step 2: Fit the Model
model = LinearRegression()
model.fit(X, Y)

# Get coefficients and intercept
b0 = model.intercept_
b1, b2 = model.coef_
print(f"Intercept (b0): {b0}")
print(f"Coefficient for X1 (b1): {b1}")
print(f"Coefficient for X2 (b2): {b2}")

# Step 3: Predictions
Y_pred = model.predict(X)

# Step 4: Evaluate the Model
print(f"Mean Squared Error: {mean_squared_error(Y, Y_pred):.2f}")
print(f"R-squared: {r2_score(Y, Y_pred):.2f}")

# Step 5: Visualize Results in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot original data points
ax.scatter(X1, X2, Y, color='red', label='Actual Data')


# Plot regression plane
X1_grid, X2_grid = np.meshgrid(np.linspace(1, 5, 10), np.linspace(2, 10, 10))
Y_grid = b0 + b1 * X1_grid + b2 * X2_grid
ax.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.5, cmap='viridis')

# Labels and Title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Multiple Linear Regression')
ax.legend()
plt.show()
