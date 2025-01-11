import numpy as np
from sklearn.linear_model import LogisticRegression

# Given data
hours_of_study = np.array([29, 15, 33, 28, 39]).reshape(-1, 1)
pass_fail = np.array([0, 0, 1, 1, 1])

# Create and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(hours_of_study, pass_fail)

# Predict the probability of passing for 33 hours of study
probability = log_reg.predict_proba([[33]])[0, 1]  # Get the probability for class 1 (pass)

# Output the probability
print(f"The probability of passing with 33 hours of study is: {probability:.4f}")
