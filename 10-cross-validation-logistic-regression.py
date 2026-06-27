"""
10 - Cross-Validation using Logistic Regression

Description:
This program demonstrates how to evaluate a Machine Learning model using
cross-validation with Logistic Regression on the Iris dataset.

What is Cross-Validation?
Cross-validation is a model evaluation technique that splits the dataset into
multiple folds. The model is trained on some folds and tested on the remaining
fold, repeating the process several times to obtain a reliable performance estimate.

Why Cross-Validation?

* Reduces overfitting
* Provides a more reliable estimate of model performance
* Utilizes the entire dataset for both training and testing

Steps Covered:

1. Load the Iris dataset
2. Create a Logistic Regression model
3. Perform 3-fold cross-validation
4. Calculate the average accuracy score
5. Perform 5-fold cross-validation
6. Compare evaluation results

Key Concepts:

* Cross-Validation
* Logistic Regression
* Model Evaluation
* Accuracy Score

"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("three Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

scores = cross_val_score(logreg, iris.data, iris.target,cv=5)
print("five Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
