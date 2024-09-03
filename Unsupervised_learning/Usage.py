from K_means import KMeans
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Assuming the KMeans class is already implemented as you shared above

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Initialize KMeans with 4 clusters (since we generated data with 4 centers)
kmeans = KMeans(k=4, max_iters=100)

# Fit the KMeans algorithm to the data
y_kmeans = kmeans.predict(X)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plotting the data points with their assigned cluster labels
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plotting the centroids
centroids = np.array(kmeans.centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()