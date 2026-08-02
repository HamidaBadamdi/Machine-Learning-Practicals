"""
Practical 36: Decision Tree Classification for Job Offer Prediction

Objective:
To build a Decision Tree Classification model that predicts whether a candidate
will receive a job offer based on CGPA, communication, aptitude, and programming skills.

Tools Used:
- Python
- Pandas
- Scikit-learn

Procedure:
1. Load the dataset.
2. Encode categorical data using LabelEncoder.
3. Train a Decision Tree Classifier.
4. Display the decision tree rules.
5. Predict the job offer for new candidate data.

Outcome:
Successfully implemented a Decision Tree classifier and predicted job offer
status based on candidate attributes.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("job_offer_dataset.csv")

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
X = df[["CGPA", "Communication", "Aptitude", "Programming_skill"]]
y = df["Job_offered"]

# -----------------------------
# Step 4: Train Decision Tree
# -----------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

# -----------------------------
# Step 5: Display Decision Tree Rules
# -----------------------------
feature_names = ["CGPA", "Communication", "Aptitude", "Programming_skill"]

print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=feature_names))

# -----------------------------
# Step 6: Test Data
# Example:
# CGPA = High
# Communication = Good
# Aptitude = High
# Programming Skill = Good
# -----------------------------
test = pd.DataFrame({
    "CGPA": ["High"],
    "Communication": ["Good"],
    "Aptitude": ["High"],
    "Programming_skill": ["Good"]
})

# Encode test data
for column in test.columns:
    test[column] = encoders[column].transform(test[column])

# -----------------------------
# Step 7: Prediction
# -----------------------------
prediction = model.predict(test)

result = encoders["Job_offered"].inverse_transform(prediction)

print("\nPrediction:")
print("Job Offered:", result[0])
