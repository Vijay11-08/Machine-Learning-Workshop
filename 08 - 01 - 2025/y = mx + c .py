import matplotlib.pyplot as plt
import numpy as np

# Step 2: Define parameters
m = 2  # Slope
c = 1  # Y-intercept

# Step 3: Choose x-values
x_values = [1, 2, 3, 4, 5]  # Specific x-values

# Step 4: Calculate y-values
y_values = [m * x + c for x in x_values]

# Step 5: Create a bar chart

plt.bar(x_values, y_values, color='skyblue')
plt.xlabel('x - Independent Variable')
plt.ylabel('y - Dependent Variable')
plt.title('Bar Chart of y = mx + c')
plt.xticks(x_values)  # Ensure all x-values are shown on the x-axis
plt.grid(axis='y', linestyle='--')  # Add horizontal grid lines
plt.show()