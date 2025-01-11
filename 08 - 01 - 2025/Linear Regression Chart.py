import matplotlib.pyplot as plt
import numpy as np

# Step 2: Define data
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

# Step 3: Calculate regression line
coefficients = np.polyfit(x, y, 1)  # 1 for linear (degree 1)
regression_line = np.poly1d(coefficients)

# Step 4: generate x values for the line
x_line = x

# Step 5: Calculate y values for the line
y_line = regression_line(x_line)

# Step 6: Plot
plt.scatter(x, y, color='red', label='Data Points')  # Scatter plot of data points
plt.plot(x_line, y_line, color='blue', label=f'Regression Line: y={coefficients[0]:.2f}x+{coefficients[1]:.2f}')  # Regression line

# Step 7: Customize
plt.xlabel('x - Independent Variable', fontsize=12)
plt.ylabel('y - Dependent Variable', fontsize=12)
plt.title('Linear Regression Chart', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # Add a horizontal line at y=0
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # Add a vertical line at x=0
plt.show()