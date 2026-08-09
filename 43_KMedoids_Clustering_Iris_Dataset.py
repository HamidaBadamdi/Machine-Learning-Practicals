"""
Practical 43: K-Medoids Clustering on Iris Dataset

Objective:
To implement K-Medoids clustering on the Iris dataset and evaluate the
clustering results using a confusion matrix and classification report.

Tools Used:
- Python
- NumPy
- Matplotlib
- Scikit-learn
- Scikit-learn-extra

Procedure:
1. Load the Iris dataset.
2. Apply K-Medoids clustering with three clusters.
3. Generate cluster labels.
4. Evaluate the clustering results.
5. Visualize the generated clusters.

Outcome:
Successfully implemented K-Medoids clustering and evaluated the clustering
results using classification metrics and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform k-medoids clustering
k = 3  # number of clusters
kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X)

# Get cluster labels
labels = kmedoids.labels_

# Evaluate clustering performance
print("Confusion Matrix:")
print(confusion_matrix(y, labels))
print("\nClassification Report:")
print(classification_report(y, labels))

# Visualize the clusters (using first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('KMedoids Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()
