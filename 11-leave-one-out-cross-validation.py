"""
11 - Leave-One-Out Cross-Validation (LOOCV)

Description:
This program demonstrates Leave-One-Out Cross-Validation (LOOCV) using
Logistic Regression on the Iris dataset.

What is Leave-One-Out Cross-Validation?
LOOCV is a special case of k-fold cross-validation where each sample is
used once as the test set while all remaining samples are used for training.

Why Use LOOCV?
- Maximizes the amount of training data
- Provides an almost unbiased estimate of model performance
- Useful for small datasets

Steps Covered:
1. Load the Iris dataset
2. Create a Logistic Regression model
3. Apply Leave-One-Out Cross-Validation
4. Calculate accuracy for each iteration
5. Compute the overall mean accuracy

Key Concepts:
- Leave-One-Out Cross-Validation (LOOCV)
- Logistic Regression
- Model Evaluation
- Generalization Performance

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

iris = load_iris()
print("Iris labels:\n{}".format(iris.target))
logreg = LogisticRegression()


loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
