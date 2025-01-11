# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Bengaluru house dataset
df = pd.read_csv('/content/drive/MyDrive/AI-ML Workshop /Bengaluru_House_Data.csv')

# Display dataset information
print("Dataset shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values by filling or dropping
df = df.dropna(subset=['price'])  # Drop rows with missing target 'price'

# Convert 'bath' and 'balcony' columns to numeric, forcing errors to NaN
df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')

# Drop rows with NaN values in 'bath' and 'balcony' columns
df = df.dropna(subset=['bath', 'balcony'])

# Convert categorical columns (e.g., location) to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop('price', axis=1)  # Features (excluding target column 'price')
y = df['price']  # Target (price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()
