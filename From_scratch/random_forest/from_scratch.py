import numpy as np 
from decision_trees import DT
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sample_split=2, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DT(max_depth=self.max_depth, 
                      min_sample_split=self.min_sample_split, 
                      n_feats=self.n_feats)
            
            # Bootstrapping: Sample with replacement
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Gather predictions from every tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds shape is (n_trees, n_samples). We need to swap them.
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # Majority vote for each sample
        y_pred = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        return np.array(y_pred)