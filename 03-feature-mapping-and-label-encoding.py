"""
03 - Feature Mapping and Class Label Encoding

Description:
This program demonstrates how to convert categorical features and class labels
into numerical values using mapping techniques and LabelEncoder.

Why This is Important?
Machine Learning models require numerical input. This practical shows:

* Manual feature mapping for ordinal data
* Class label encoding
* Reverse mapping to original values

Steps Covered:

1. Load dataset using pandas
2. Apply feature mapping for ordinal values (size)
3. Perform reverse mapping
4. Encode class labels using dictionary mapping
5. Decode class labels back to original values
6. Apply LabelEncoder for automatic encoding
7. Reverse transformation using LabelEncoder

"""



# Feature and Class label mapping 
import pandas as pd
import numpy as np
df=pd.read_csv('D:/4121/ML/tempbook.csv')
print(df)

#feature mapping with numerical value
size_mapping={'XL':3,'L':2,'M':1}
df['size']=df['size'].map(size_mapping)
print(df)

#reverse mapping of size
inv_size_mapping={v: k for k, v in size_mapping.items()}
df['size']=df['size'].map(inv_size_mapping)
print(df)

#class label mapping
class_mapping={label:idx for idx, label in enumerate(np.unique(df['class']))}
print(class_mapping)
df['class']= df['class'].map(class_mapping)
print(df)


#reverse mapping of class
inv_class_mapping={v:k for k, v in class_mapping.items()}
df['class']= df['class'].map(inv_class_mapping)
print(df)

#with the help of labelencoder
from sklearn.preprocessing import LabelEncoder
class_le= LabelEncoder()
y=class_le.fit_transform(df['class'].values)
print(y)

#reverse
class_le.inverse_transform(y)
print(y)

