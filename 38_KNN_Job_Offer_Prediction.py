"""
Practical 38: K-Nearest Neighbors (KNN) Classification for Job Offer Prediction

Objective:
To implement a KNN classifier to predict whether a candidate will receive
a job offer based on CGPA, communication, aptitude, and programming skills.

Tools Used:
- Python
- Pandas
- Scikit-learn

Procedure:
1. Load the dataset.
2. Encode categorical data using LabelEncoder.
3. Train a KNN classifier.
4. Predict the job offer for new candidate data.

Outcome:
Successfully implemented a KNN classification model to predict job offer
status using candidate attributes.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

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
# Step 4: Train KNN Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# -----------------------------
# Step 5: Test Data
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
# Step 6: Prediction
# -----------------------------
prediction = model.predict(test)

result = encoders["Job_offered"].inverse_transform(prediction)

print("\nPrediction:")
print("Job Offered :", result[0])
