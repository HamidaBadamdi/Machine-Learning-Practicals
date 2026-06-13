"""
01 - Label Encoding Practical

Description:
This program demonstrates how to convert categorical data into numerical form
using LabelEncoder from sklearn.preprocessing.

Why Label Encoding?
Machine Learning models cannot work with categorical (text) data directly.
Label Encoding helps in converting each category into a unique numeric value.

Steps Covered:

Define categorical data
Apply LabelEncoder
Transform categories into numeric labels
Reverse the encoded data back to original form

Author: Your Name
"""
from sklearn.preprocessing import LabelEncoder

# Sample categorical data
categories = ['red', 'blue', 'green', 'yellow', 'red', 'blue', 'green']

# Creating a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the categorical data
encoded_data = label_encoder.fit_transform(categories)

# Displaying the original and encoded data
print("Original categories:", categories)
print("Encoded data:", encoded_data)

# If you want to inverse transform the encoded data back to original categories
decoded_data = label_encoder.inverse_transform(encoded_data)
print("Decoded data:", decoded_data)
