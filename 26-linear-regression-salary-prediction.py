"""
26 - Salary Prediction using Linear Regression

Description:
This program demonstrates Simple Linear Regression using an Experience
vs Salary dataset. The model predicts salary based on years of
experience and visualizes the regression line.

Key Concepts:
- Simple Linear Regression
- Salary Prediction
- Train-Test Split
- Prediction
- Data Visualization

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv("D:/Winter 2023/BSc DS 4/Programs/expvssal.csv")

x=data.iloc[:,0:1]
y=data.iloc[:,1]

print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train,y_train)

# Making predictions
y_pred_test = model.predict(X_test)     # predicted value of y_test
y_pred_train = model.predict(X_train)   # predicted value of y_train

y_pred=model.predict([[11]])

print("predicted salary is:",y_pred)

# Plotting results
plt.scatter(X_train, y_train, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'exp/sal', loc='best', facecolor='white')
plt.box(False)
plt.show()
