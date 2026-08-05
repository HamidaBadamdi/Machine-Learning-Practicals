"""
Practical 39: K-Means Clustering on Breast Cancer Dataset

Objective:
To implement the K-Means clustering algorithm for grouping breast cancer data
into clusters based on their feature similarities.

Tools Used:
- Python
- Pandas
- Scikit-learn
- Matplotlib

Procedure:
1. Load the Breast Cancer dataset.
2. Standardize the feature values.
3. Apply the K-Means clustering algorithm.
4. Visualize the generated clusters.

Outcome:
Successfully grouped the breast cancer dataset into clusters and visualized
the clustering results using a scatter plot.

"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(df['mean radius'], df['mean texture'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('K-means Clustering of Breast Cancer Data')
plt.show()
