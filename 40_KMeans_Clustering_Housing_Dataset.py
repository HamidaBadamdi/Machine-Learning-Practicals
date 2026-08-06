"""
Practical 40: K-Means Clustering on Housing Dataset

Objective:
To apply the K-Means clustering algorithm on a housing dataset and evaluate
the clustering performance using the Silhouette Score.

Tools Used:
- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

Procedure:
1. Load the housing dataset.
2. Normalize the selected features.
3. Train a K-Means clustering model.
4. Visualize the generated clusters.
5. Evaluate clustering performance using the Silhouette Score.

Outcome:
Successfully clustered housing data into groups and assessed the quality of
the clusters using visualization and the Silhouette Score.
"""
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 


home_data = pd.read_csv('datasets/housing_dataset.csv', usecols = ['longitude', 'latitude', 'house_value'])
home_data.head()

sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'house_value')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['house_value']], test_size=0.33, random_state=0)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.show()

sns.boxplot(x = kmeans.labels_, y = y_train['house_value'])
plt.show()


print("silhouette score is",(silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')))
      
