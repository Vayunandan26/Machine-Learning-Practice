import numpy as np 
import matplotlib.pyplot as plt

def euclidean_dist(x, y):
    # FIXED: Square the difference BEFORE summing
    return np.sqrt(np.sum((x - y)**2))

class KMeans():
    def __init__(self, k=5, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        # We initialize these as empty, they get filled in predict/fit
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        
    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape
        
        # Initialize centroids by picking random samples
        random_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = self.x[random_idx]

        for _ in range(self.max_iters):
            # FIXED: Assign samples to the closest centroids
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()
                
            # FIXED: Calculate new centroids from the clusters
            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # Check for convergence (if centroids stopped moving)
            if self._is_converged(centroid_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        # FIXED: Array size should be n_samples, not n_features
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels        
        
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample, centroids)
            # FIXED: Append the index to the correct sub-list
            clusters[centroid_idx].append(idx)
        return clusters 
            
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_dist(sample, point) for point in centroids]
        return np.argmin(distances)
    
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_mean = np.mean(self.x[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
            else:
                # If a cluster is empty, keep the centroid where it was
                centroids[cluster_idx] = self.x[np.random.choice(self.n_samples)]
        return centroids
    
    def _is_converged(self, centr_old, centr):
        dist = [euclidean_dist(centr_old[i], centr[i]) for i in range(self.k)]
        return sum(dist) < 1e-4 
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            if len(index) > 0:
                # FIXED: This now correctly slices the data points
                point = self.x[index].T
                ax.scatter(*point)
            
        for point in self.centroids:
            ax.scatter(*point, marker='X', color='black', linewidth=2)
            
        plt.show()