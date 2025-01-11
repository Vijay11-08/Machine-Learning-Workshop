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

# Check for missing values in the dataset
print("\nMissing values before filling:")
print(df.isnull().sum())

# Handle missing values:
# For categorical columns (e.g., 'location', 'area_type', 'availability', 'society'), use mode
df['location'] = df['location'].fillna(df['location'].mode()[0])
df['area_type'] = df['area_type'].fillna(df['area_type'].mode()[0])
df['availability'] = df['availability'].fillna(df['availability'].mode()[0])
df['society'] = df['society'].fillna(df['society'].mode()[0])

# For numerical columns, use median
df['size'] = df['size'].fillna(df['size'].mode()[0])  # 'size' seems categorical but has missing values, so we fill with mode
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())

# Check for missing values after filling
print("\nMissing values after filling:")
print(df.isnull().sum())

# Convert categorical columns to dummy variables
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
