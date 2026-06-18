"""
06 - Handling Missing Values

Description:
This program demonstrates how to detect and handle missing values (NaN)
in a dataset using pandas.

Why Handling Missing Values?
Missing data can affect model performance. It is important to:

* Detect missing values
* Remove or handle them appropriately

Steps Covered:

1. Create dataset with missing values (NaN)
2. Detect missing values using isnull()
3. Drop rows with missing values
4. Drop columns with missing values
5. Drop rows where all values are NaN
6. Use threshold to retain rows with minimum valid values
7. Drop rows based on specific column conditions

Key Functions Used:

* isnull()
* dropna()
"""


import numpy as np
import pandas as pd
from io import StringIO

# NaN (Not a Number) represents missing or undefined data
narr=np.array([[1,2,3,4],[np.nan,6,7,8],[9,10,np.nan,12]])
print(narr)

df = pd.DataFrame(narr, columns=['A','B','C','D'])
print ('-----1------')
print (df.isnull().sum())
print ('-----2------')
print (df.values)
print ('-----3------')

#Drop rows with any NaN values
print (df.dropna())
print ('-----4------')
#Drop columns with any NaN values:
print (df.dropna(axis=1))
print ('-----5------')
#Drop rows where all values are NaN:
print (df.dropna(how='all'))
print ('-----6------')
#thresh: This parameter is an integer that represents the minimum number of
#non-null values required to keep a row or column. If the number of non-null
#values is less than the specified threshold, the row or column is dropped.
print (df.dropna(thresh=4))
print ('-----7------')
#The subset parameter allows you to specify a subset of columns or rows to consider 
#when performing the operation. You can use subset to drop NaN values only in specific columns or rows.
print (df.dropna(subset=['A']))
print ('------8------')
#Drop columns where all values are NaN:
print (df.dropna(axis=1,how='all'))
