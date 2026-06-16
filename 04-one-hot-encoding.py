"""
04 - One Hot Encoding Practical

Description:
This program demonstrates how to convert categorical data into numerical
format using One-Hot Encoding.

Why One-Hot Encoding?
Unlike Label Encoding, One-Hot Encoding avoids assigning ordinal relationships
between categories by creating separate binary columns for each category.

Steps Covered:

1. Load dataset using pandas
2. Apply LabelEncoder for initial transformation
3. Use ColumnTransformer with OneHotEncoder (sklearn)
4. Create dummy variables using pandas get_dummies
5. Compare both approaches

"""



# Onehotencoder for mapping
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df=pd.read_csv('D:/4121/ML/tempbook.csv')
print(df)

x=df[['color','size','price']].values
color_le=LabelEncoder()
x[:,0]=color_le.fit_transform(x[:,0])
print(x)


#One way with onehotencoder for creating dummy features with sklearn
ct = ColumnTransformer([("color", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
print(x)

#Other way with pandas
print(pd.get_dummies(df[['color','price']]))
