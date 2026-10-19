import numpy as np 
import matplotlib.pyplot as plt

def euclidean_dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

class KMeans():
    def __init__(self, k=5, max_iters=1000, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        
    def predict(self, x):
        self.x = x
        self.n_samples , self.n_features = x.shape
        
        # initialize centroids. 
        random_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[idx] for idx in random_idx]
        #optimization. 
        for _ in range(self.max_iters):
            #update the clusters. 
            self.clusters = self.create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
                
            #update centroids 
            centroid_old = self.centroids
            self.centroids = self.get_centroid(self.clusters)
            if self.plot_steps:
                self.plot()
            #check for convergence 
            if self.converged(centroid_old, self.centroids):
                break
        #return cluster labels 
        return self.cluster_labels(self.clusters)
    
    def cluster_labels(self, cluster):
        labels = np.empty(self.n_samples)
        for cluster_idx, clust in enumerate(cluster):
            for idx in clust:
                labels[idx] = cluster_idx
                
        return labels        
        
    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self.closest_centroids(sample, centroids)
            clusters[centroid_idx].append(centroid_idx)
        return clusters 
            
    def closest_centroids(self, sample, centroids):
        dist = [euclidean_dist(sample, point) for point in centroids]
        closest_dist = np.argmin(dist)
        return closest_dist
    
    def get_centroid(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for clust_idx, clust in enumerate(clusters):
            clust_mean = np.mean(self.x[clust], axis=0)
            centroids[clust_idx] = clust_mean            
        return centroids
    
    def converged(self, centr_old, centr):
        dist = [euclidean_dist(centr_old[i], centr[i]) for i in range(self.k)]
        return sum(dist) < 1e-6
    
    
    def plot(self):
        
        fig, ax = plt.subplots(figsize =(12,8))
        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)
            
        for point in self.centroids:
            ax.scatter(*point, marker='X', color='black', linewidth=2)
            
        plt.show()