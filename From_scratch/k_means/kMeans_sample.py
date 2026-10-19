import numpy as np 
from sklearn import datasets 
from from_scratch import KMeans

x, y = datasets.make_blobs(n_samples=10000, centers=4, n_features=2, cluster_std=0.60, random_state=12)

clusters = len(np.unique(y))

kmeans = KMeans(k=clusters, plot_steps=True)
y_pred = kmeans.predict(x)

kmeans.plot()