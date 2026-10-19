import numpy as np 
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p*np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 
        
    def is_leaf_node(self):
        return self.value is not None 
    
class DT:
    def __init__(self, min_sample_split = 10, max_depth = 100, n_feats = 30):
        self.min_sample_split = min_sample_split 
        self.max_depth = max_depth 
        self.n_feats = n_feats
        self.root = None 
        
    def fit(self, x, y):
        self.n_feats = x.shape[1] if not self.n_feats else min(self.n_feats, x.shape[1])
        self.root = self.grow_tree(x,y)
        
    def grow_tree(self,x,y, depth = 0):    
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))
        
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split ):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idc = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh = self.best_criteria(x, y, feat_idc)
        left_idc, right_idc = self.split(x[:, best_feat], best_thresh)
        left = self.grow_tree(x[left_idc, :], y[left_idc], depth + 1)
        right = self.grow_tree(x[right_idc, :], y[right_idc], depth + 1)
        return Node(best_feat, best_thresh, left, right)
        
    def best_criteria(self, x, y, feat_idx):
        best_gain = -1
        split_idx, split_thresh = None, None 
        for idc in feat_idx:
            x_column = x[: ,idc]
            threshold = np.unique(x_column)
            
            for thr in threshold:
                gain = self.info_gain(y, x_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idc 
                    split_thresh = thr 
                    
        return split_idx, split_thresh
        
    def info_gain(self,y, x, split_thresh):
        parent_entropy = entropy(y)
        left_idc, right_idc = self.split(x, split_thresh)
        if len(left_idc) ==0 or len(right_idc) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idc), len(right_idc)
        e_l, e_r = entropy(y[left_idc]), entropy(y[right_idc])
        child_entropy = (n_l/ n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy
        
    def split(self, x, thr):
        left_idc = np.argwhere(x <= thr).flatten()
        right_idc = np.argwhere(x > thr).flatten()
        return left_idc, right_idc
    
    def most_common_label(self, y):
        coun = Counter(y)
        most_common = coun.most_common(1)[0][0]
        return most_common
    
    def traverse(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)
    
    def predict(self,x):
        return np.array([self.traverse(x_, self.root) for x_ in x ])
    
        