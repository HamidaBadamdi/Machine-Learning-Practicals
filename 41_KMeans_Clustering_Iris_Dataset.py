"""
Practical 41: K-Means Clustering on Iris Dataset

Objective:
To implement the K-Means clustering algorithm on the Iris dataset and evaluate
the clustering results using a confusion matrix and classification report.

Tools Used:
- Python
- NumPy
- Matplotlib
- Scikit-learn

Procedure:
1. Load the Iris dataset.
2. Apply the K-Means clustering algorithm.
3. Generate cluster labels.
4. Evaluate clustering using a confusion matrix and classification report.
5. Visualize the clusters and cluster centroids.

Outcome:
Successfully clustered the Iris dataset, evaluated the clustering performance,
and visualized the clusters along with their centroids.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform k-means clustering
k = 3  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

# Get cluster labels
labels = kmeans.labels_

# Evaluate clustering performance
print("Confusion Matrix:")
print(confusion_matrix(y, labels))
print("\nClassification Report:")
print(classification_report(y, labels))

# Visualize the clusters (using first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
