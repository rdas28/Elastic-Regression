# ElasticNet Regression

## Table of Contents:
Overview
Features
Model Details
Objective Function
Example Code
Configurable Parameters
Limitations
Contributors

## Overview:
This repository includes a custom implementation of ElasticNet Regression built entirely from scratch using only NumPy and pandas. No external libraries like scikit-learn or TensorFlow are utilized, making it an excellent resource for understanding how ElasticNet works under the hood. The model is optimized using gradient descent, providing a hands-on approach to its mechanics.

ElasticNet Regression combines the strengths of L1 (Lasso) and L2 (Ridge) regularization, making it effective for feature selection and handling correlated predictors. This implementation also demonstrates how optimization techniques can be fine-tuned.

## Features:
Custom ElasticNet Implementation: Combines L1 and L2 regularization for robust linear regression.
Manual Gradient Descent Optimization: Allows full control over the training process by implementing gradient descent from scratch.

## Model Details:
ElasticNet Regression is a hybrid regularization technique that leverages both L1 and L2 penalties. It is particularly useful for feature selection and addressing multicollinearity among predictors.

## Objective Function:
The ElasticNet model minimizes the following objective function:

from elasticnet import ElasticNetModel

# Initialize the ElasticNet model
model = ElasticNetModel(
    alpha=1.0, 
    l1_ratio=0.5, 
    max_iter=2000, 
    convergence_criteria=1e-4, 
    step_size=0.005, 
    bias_term=True
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

## Example Code:

# Train the model
outcome = model.fit(X_train, y_train)

# Make predictions
y_pred = outcome.predict(X_test)

# Evaluate the model
r2 = outcome.r2_score(y_test, y_pred)
rmse = outcome.rmse(y_test, y_pred)

print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")


## Configurable Parameters:
The ElasticNet model provides the following adjustable parameters:

alpha: Controls the regularization strength (default: 1.0).
l1_ratio: Defines the balance between L1 (Lasso) and L2 (Ridge) regularization (default: 0.5).
max_iter: Sets the maximum number of iterations for gradient descent (default: 2000).
convergence_criteria: Stopping criterion for training when the difference between iterations is below this threshold (default: 1e-4).
step_size: Determines the learning rate for gradient descent (default: 0.005).
bias_term: Whether to include an intercept in the model (default: True).

## Limitations:
Slow Convergence: The model may take longer to converge when dealing with large datasets or highly correlated features. Employing alternate optimization techniques like coordinate descent can improve convergence speed.
Precision Issues: Gradient descent might not achieve the same precision as closed-form solutions, especially for highly sensitive applications.

## Contributors:
Riddhi Das (A20582829): rdas8@hawk.iit.edu
Madhur Gusain (A20572395): mgusain@hawk.iit.edu



















