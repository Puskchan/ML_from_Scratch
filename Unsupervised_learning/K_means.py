import numpy as np

def eu_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self,k=5,max_iters=100,plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample]

        for _ in range(self.max_iters):

            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroid(self.clusters)

            if self._is_converged(centroids_old,self.centroids):
                break

        return self._get_cluster_labels(self.clusters)
    

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self,sample, centroids):
        distances = [eu_dist(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _get_centroid(self,clusters):
        centroids = np.zeros((self.k,self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self,centroids_old, centroids):
        distances = [eu_dist(centroids_old[i],centroids[i]) for i in range(self.k)]
        return sum(distances) == 0