import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])  # Actual values
y_pred = np.array([8, 11, 14, 17, 20])  # Predicted values

# Calculate squared errors
squared_errors = (y - y_pred)**2
mse = np.mean(squared_errors)


# Plot the squared errors as discrete points
plt.figure(figsize=(10, 6))
plt.scatter(x, squared_errors, color='blue', label='Squared Errors', s=100)  # Markers for errors
plt.axhline(y=mse, color='red', linestyle='--', label=f'MSE = {mse:.2f}')  # MSE as reference
plt.title('Loss Function Visualization (Discrete Squared Errors)')
plt.xlabel('x (Input Data)')
plt.ylabel('Squared Error')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
