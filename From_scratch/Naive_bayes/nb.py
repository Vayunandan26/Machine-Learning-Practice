import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 
from from_scratch import NB

x, y = datasets.make_classification(n_samples=10000, n_features=15, n_classes=2, random_state=55)

def train_test_split(x, y, split = 0.2, random = None):
    
    if random is not None :
        np.random.seed(random)
    
    assert len(x) == len(y)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    split_count = int(len(x)*split)
    
    train_indices = indices[split_count:]
    test_indices = indices[:split_count]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices] 

x_train, x_test, y_train, y_test = train_test_split(x, y, split= 0.2, random = 42)

model = NB()
model.fit(x_train, y_train)

predict = model.predict(x_test)

def accuracy(y_true, pred):
    acc = np.sum(y_true == pred ) / len(y_true)
    
    return acc

accuracyy = accuracy(y_test, predict)

print(accuracyy)