import numpy as np 
from sklearn import datasets
from from_scratch import RandomForest

br = datasets.load_breast_cancer()
x = br.data
y = br.target

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

x_train, x_test, y_train, y_test = train_test_split(x, y, split = 0.2, random = 123)

model = RandomForest()
model.fit(x_train, y_train)

predict = model.predict(x_test)

def accuracy(y, pred):
    return np.sum(y == pred)/len(y)

acc = accuracy(y_test, predict)

print(acc)