"""
Practical 37: K-Nearest Neighbors (KNN) Classification for Buys Computer Prediction

Objective:
To implement a K-Nearest Neighbors (KNN) classifier to predict whether a customer
will buy a computer based on age, income, student status, and credit rating.

Tools Used:
- Python
- Pandas
- Scikit-learn

Procedure:
1. Load the dataset.
2. Encode categorical data using LabelEncoder.
3. Train a KNN classifier.
4. Predict the output for new customer data.

Outcome:
Successfully implemented a KNN classification model to predict computer purchase
decisions using categorical features.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("datasets/buys_computer.csv")

print("Original Dataset:\n")
print(df)

# -----------------------------
# Step 2: Encode Categorical Data
# -----------------------------
encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print("\nEncoded Dataset:\n")
print(df)

# -----------------------------
# Step 3: Separate Features & Target
# -----------------------------
X = df[["Age", "Income", "Student", "CreditRating"]]
y = df["BuysComputer"]

# -----------------------------
# Step 4: Train KNN Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# -----------------------------
# Step 5: Test Data
# -----------------------------
test = pd.DataFrame({
    "Age": ["Youth"],
    "Income": ["Medium"],
    "Student": ["Yes"],
    "CreditRating": ["Fair"]
})

# Encode test data
for column in test.columns:
    test[column] = encoders[column].transform(test[column])

# -----------------------------
# Step 6: Prediction
# -----------------------------
prediction = model.predict(test)

result = encoders["BuysComputer"].inverse_transform(prediction)

print("\nPrediction:")
print("Will the person buy a computer? :", result[0])
