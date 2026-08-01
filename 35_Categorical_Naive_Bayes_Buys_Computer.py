import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Step 1: Load dataset from CSV file
df = pd.read_csv('buys_computer.csv')

print('Original Dataset:\\n')
print(df)

# Step 2: Encode categorical data
encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print('\\nEncoded Dataset:\\n')
print(df)

# Step 3: Separate features and target
X = df[['Age', 'Income', 'Student', 'CreditRating']]
y = df['BuysComputer']

# Step 4: Train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)
# Step 5: Test data
# Example:
# Age = Youth
# Income = Medium
# Student = Yes
# CreditRating = Fair

test_data = pd.DataFrame({
    'Age': ['Youth'],
    'Income': ['Medium'],
    'Student': ['Yes'],
    'CreditRating': ['Fair']
})

# Encode test data
for column in test_data.columns:
    test_data[column] = encoders[column].transform(test_data[column])

# Step 6: Predict
prediction = model.predict(test_data)

# Decode prediction
result = encoders['BuysComputer'].inverse_transform(prediction)

print('\\nPrediction:')
print('Will the person buy a computer? ->', result[0])
