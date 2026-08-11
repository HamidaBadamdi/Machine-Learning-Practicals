"""
Practical 44: K-Nearest Neighbors Classification on Breast Cancer Dataset

Objective:
To implement a KNN classifier for classifying breast cancer tumors as
malignant or benign and evaluate the model using standard classification metrics.

Tools Used:
- Python
- Scikit-learn

Procedure:
1. Load the Breast Cancer dataset.
2. Split the data into training and testing sets.
3. Train a KNN classifier with k=5.
4. Predict the test data.
5. Evaluate the model using accuracy, confusion matrix, and classification report.

Outcome:
Successfully implemented KNN classification and evaluated its performance
for breast cancer tumor classification.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# Step 1: Load Breast Cancer Dataset
# ----------------------------------
cancer = load_breast_cancer()

X = cancer.data        # Features
y = cancer.target      # Target (0 = Malignant, 1 = Benign)

print("Dataset Shape:", X.shape)
print("Number of Features:", len(cancer.feature_names))
print("Target Names:", cancer.target_names)

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
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
