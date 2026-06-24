"""
08 - Decision Tree and Feature Importance

Description:
This program demonstrates how to train a Decision Tree Classifier
and analyze feature importance using the Iris dataset.

What is Decision Tree?
A Decision Tree is a supervised learning algorithm used for classification
and regression tasks. It splits data based on feature values.

Why Feature Importance?
Feature importance helps identify which features contribute most
to the prediction, improving model understanding.

Steps Covered:

1. Load dataset (Iris dataset)
2. Split data into training and testing sets
3. Train Decision Tree model
4. Extract feature importance
5. Display importance of each feature

Key Concepts:

* Supervised Learning
* Classification
* Feature Importance

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Get feature importances
feature_importances = clf.feature_importances_

# Print feature importances
for feature, importance in zip(iris.feature_names, feature_importances):
    print(f"{feature}: {importance}")
