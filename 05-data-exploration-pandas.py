"""
05 - Data Exploration using Pandas

Description:
This program demonstrates how to explore and understand a dataset using pandas.

Why Data Exploration?
Before applying any Machine Learning model, it is important to understand:

* Structure of the dataset
* Data types
* Size and dimensions
* Basic information about features

Steps Covered:

1. Load dataset from CSV file
2. Display dataset content
3. View first few rows using head()
4. Check shape (rows and columns)
5. Display column names
6. Check data types of each column
7. Get dataset dimensions and size
8. Load dataset from Excel file

"""

import pandas as pd

winedatac=pd.read_csv('datasets/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)

winedatae=pd.read_excel('datasets/wine.xlsx')

print('\n')
print(windedatae)



