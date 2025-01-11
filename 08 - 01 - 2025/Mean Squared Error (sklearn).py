import numpy as np
from sklearn.linear_model import LinearRegression

# Given data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Reshaping for sklearn
y = np.array([5, 7, 9, 11, 13])

# Step 1: Fit a linear regression model
model = LinearRegression()
model.fit(x, y)


# Predicted values
y_pred = model.predict(x)

# Step 2: Calculate MSE manually
mse_manual = np.mean((y - y_pred)**2)
print(f"Mean Squared Error (Manual): {mse_manual}")

# Step 3: Verify using sklearn
from sklearn.metrics import mean_squared_error
mse_sklearn = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (sklearn): {mse_sklearn}")
