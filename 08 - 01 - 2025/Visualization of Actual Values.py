import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])  # Actual values
y_pred = np.array([8, 11, 14, 17, 20])  # Predicted values

# Calculate squared errors
squared_errors = (y - y_pred)**2
mse = np.mean(squared_errors)

# Plot actual points
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='green', label='Actual Values (y)', s=100, marker='o')

# Plot predicted points
plt.scatter(x, y_pred, color='orange', label='Predicted Values ($\hat{y}$)', s=100, marker='x')

# Plot squared errors
plt.scatter(x, squared_errors, color='blue', label='Squared Errors', s=100, marker='s')

# MSE reference line
plt.axhline(y=mse, color='red', linestyle='--', label=f'MSE = {mse:.2f}')

# Chart details
plt.title('Visualization of Actual Values, Predictions, and Errors', fontsize=14)
plt.xlabel('x (Input Data)', fontsize=12)
plt.ylabel('y / Errors', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
