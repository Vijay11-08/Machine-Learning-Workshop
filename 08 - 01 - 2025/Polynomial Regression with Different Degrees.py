import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 6, 12, 20, 30])

# Degrees to compare
degrees = [1, 2, 3]

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', label='Data Points')

for degree in degrees:
    # Transform features to polynomial
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)

    # Fit model
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)


    # Plot polynomial fit
    plt.plot(x, y_pred, label=f'Degree {degree} (MSE: {mean_squared_error(y, y_pred):.2f})')

plt.title('Polynomial Regression with Different Degrees')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
