import numpy as np

# Data
x = np.array([2, 3, 4, 5, 6])
y = np.array([10, 14, 18, 22, 26])

# Hyperparameters

alpha = 0.01  # Learning rate
iterations = 1000
m = 0  # Initial slope
c = 0  # Initial intercept

# Gradient Descent
n = len(x)
for _ in range(iterations):
    y_pred = m * x + c
    dm = -(2/n) * np.sum(x * (y - y_pred))  # Gradient for m
    dc = -(2/n) * np.sum(y - y_pred)       # Gradient for c
    m -= alpha * dm                        # Update m
    c -= alpha * dc                        # Update c

print(f"Slope (m): {m:.2f}, Intercept (c): {c:.2f}")
