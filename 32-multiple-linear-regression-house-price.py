"""
32 - House Price Prediction using Multiple Linear Regression

Description:
This program builds a Multiple Linear Regression model to predict
house prices based on square feet and number of bedrooms.
The model is evaluated using MAE, MSE, RMSE, and R² score.

Key Concepts:
- Multiple Linear Regression
- Train-Test Split
- House Price Prediction
- Regression Evaluation Metrics

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_excel("../datasets/Multivariate Linear Regression.xlsx")

# Display dataset
print(df.head())

# Features and Target
X = df[["Square Feet", "Number of Bed Rooms"]]
y = df["Price of House"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Model parameters
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Test predictions
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# Predict for new data
square_feet = float(input("\nEnter Square Feet: "))
bedrooms = int(input("Enter Number of Bedrooms: "))

predicted_price = model.predict([[square_feet, bedrooms]])

print("\nPredicted House Price:", predicted_price[0])

# Model Evaluation
print("\nModel Evaluation")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
