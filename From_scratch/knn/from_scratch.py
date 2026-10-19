import numpy as np 
from collections import Counter

def euclidian(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2)**2))
    return dist 

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, X):
        predict_labels = [self.pre_predict(x) for x in X]
        return np.array(predict_labels)
    
    def pre_predict(self, x):
        dist = [euclidian(x, x_tra) for x_tra in self.x_train]
        k_id = np.argsort(dist)[:self.k]
        k_labels = [self.y_train[i] for i in k_id]
        
        majority = Counter(k_labels).most_common(1)[0][0]
        return majority