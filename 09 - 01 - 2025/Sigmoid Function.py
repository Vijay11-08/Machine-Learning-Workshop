import numpy as np
import matplotlib.pyplot as plt

# Given x values
x = np.array([-9, -8, 0, 8, 9])

# Calculate z = 5x + 10
z = 5 * x + 10

# Apply Sigmoid function to z
sigmoid = 1 / (1 + np.exp(-z))

# Print the results
print("x values:", x)
print("z values (5x + 10):", z)
print("Sigmoid values:", sigmoid)


# Plotting the sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x, sigmoid, marker='o', color='blue', label='Sigmoid Function')
plt.title('Sigmoid Function (Logistic Function) for z = 5x + 10')
plt.xlabel('x')
plt.ylabel('Sigmoid(z)')
plt.grid(True)
plt.legend()
plt.show()
