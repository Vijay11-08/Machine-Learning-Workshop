import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])  # Actual values
y_pred = np.array([8, 11, 14, 17, 20])  # Predicted values

# Calculate squared errors
squared_errors = (y - y_pred)**2
mse = np.mean(squared_errors)

# Plot the squared errors
plt.figure(figsize=(10, 6))
plt.plot(x, squared_errors, marker='o', label='Squared Errors', color='blue')
plt.axhline(y=mse, color='red', linestyle='--', label=f'MSE = {mse:.2f}')
plt.title('Loss Function Visualization (Squared Errors)')
plt.xlabel('x (Input Data)')
plt.ylabel('Squared Error')
plt.legend()

plt.grid(True)
plt.show()
