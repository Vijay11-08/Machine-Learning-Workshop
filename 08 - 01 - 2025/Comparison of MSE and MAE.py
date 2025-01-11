import numpy as np
import matplotlib.pyplot as plt

# Generate data
y_true = 3
y_pred = np.linspace(0, 5, 100)

# Calculate loss
mse = (y_pred - y_true)**2
mae = np.abs(y_pred - y_true)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_pred, mse, label='MSE (Mean Squared Error)', color='blue')
plt.plot(y_pred, mae, label='MAE (Mean Absolute Error)', color='red')
plt.xlabel('Predicted Value (y_pred)')
plt.ylabel('Loss')
plt.title('Comparison of MSE and MAE')
plt.legend()

plt.show()
