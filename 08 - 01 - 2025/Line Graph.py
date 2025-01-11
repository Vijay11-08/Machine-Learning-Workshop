import matplotlib.pyplot as plt
import numpy as np

# Step 2: Define parameters
m = 2  # Slope
c = 1  # Y-intercept

# Step 3: Generate x-values
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10

# Step 4: Calculate y-values
y = m * x + c

# Step 5: Plot the graph
plt.plot(x, y, label=f'y = {m}x + {c}', color='blue', linewidth=2)

# Step 6: Customize the chart
plt.xlabel('x - Independent Variable', fontsize=12)
plt.ylabel('y - Dependent Variable', fontsize=12)
plt.title('Line Graph of y = mx + c', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 10])  # Set the x-axis limits
plt.ylim([0, 25])  # Set the y-axis limits
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5) # Add a horizontal line at y=0
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5) # Add a vertical line at x=0
plt.show()