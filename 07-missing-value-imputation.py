"""
07 - Missing Value Imputation

Description:
This program demonstrates how to handle missing values using SimpleImputer
from sklearn.

Why Imputation?
Instead of removing missing data, we can replace it with meaningful values
to preserve dataset size and improve model performance.

Steps Covered:

1. Create dataset with missing values (NaN)
2. Apply mean imputation
3. Apply median imputation
4. Apply constant value imputation
5. Compare results of different strategies

Key Concepts:

* Mean Imputation
* Median Imputation
* Constant Imputation

"""


import numpy as np
import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer

# Imputation replaces missing values instead of removing data
narr=np.array([[1,2,3,4],[np.nan,6,7,8],[9,10,np.nan,12]])
print(narr)

df = pd.DataFrame(narr, columns=['A','B','C','D'])

print('---------imputed data with mean---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#The fit method is used to compute the imputation strategy based #on the provided data X

imputer=imputer.fit(df)

#The transform method is used to apply the imputation strategy #learned during the fit phase to fill in the missing values in #the input data X.

imputed_data=imputer.transform(df)
print(imputed_data)

print('---------imputed data with median---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer=imputer.fit(df)
imputed_data=imputer.transform(df)
print(imputed_data)


print('---------imputed data with constant---------')
imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
imputer=imputer.fit(df)
imputed_data=imputer.transform(df)
print(imputed_data)



