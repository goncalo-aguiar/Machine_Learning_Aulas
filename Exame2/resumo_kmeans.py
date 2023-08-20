import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
db_index = davies_bouldin_score(X, labels)

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(3):
    ax.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i+1}")

ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidth=3, color='black', label='Centroids')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

x_min, x_max = np.min(centroids[:, 0]), np.max(centroids[:, 0])
y_min, y_max = np.min(centroids[:, 1]), np.max(centroids[:, 1])
x_range = x_max - x_min
y_range = y_max - y_min
x_margin = 0.1 * x_range
y_margin = 0.1 * y_range
ax.set_xlim([x_min - x_margin-5, x_max + x_margin+5])
ax.set_ylim([y_min - y_margin-5, y_max + y_margin+5])

print(f"Davies-Bouldin index: {db_index}")
plt.show()