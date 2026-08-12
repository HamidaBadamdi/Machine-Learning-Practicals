"""
Practical 45: K-Nearest Neighbors Classification on Iris Dataset

Objective:
To implement a KNN classifier for classifying Iris flowers into their respective
species and evaluate the model using standard classification metrics.

Tools Used:
- Python
- Scikit-learn

Procedure:
1. Load the Iris dataset.
2. Split the data into training and testing sets.
3. Train a KNN classifier with k=5.
4. Predict the test data.
5. Evaluate the model using accuracy, confusion matrix, and classification report.

Outcome:
Successfully implemented KNN classification and evaluated its performance
for Iris flower classification.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# Step 1: Load Iris Dataset
# ----------------------------------
iris = load_iris()

X = iris.data          # Features
y = iris.target        # Target

print("Dataset Shape:", X.shape)
print("Number of Features:", len(iris.feature_names))
print("Target Names:", iris.target_names)

# ----------------------------------
# Step 2: Split Dataset
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# Step 3: Train KNN Classifier
# ----------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# ----------------------------------
# Step 4: Prediction
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Step 5: Accuracy
# ----------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ----------------------------------
# Step 6: Confusion Matrix
# ----------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# Step 7: Classification Report
# ----------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
