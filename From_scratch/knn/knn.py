import numpy as np 
import matplotlib.pyplot as plt
from from_scratch import KNN
from sklearn import datasets

wine = datasets.load_wine()
x, y = wine.data, wine.target

def train_test_split(x, y, test_split = 0.2, random_state = 42):
    if random_state is not None:
        np.random.seed(random_state)
        
    assert len(x) == len(y)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    split = int(len(x)*test_split)
    
    train_indices = indices[split :]
    test_indices = indices[: split]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_split= 0.2)

for i in range(1,6) :
    model = KNN(k = i)
    model.fit(x_train, y_train)

    predict = model.predict(x_test)

    acc = np.sum(y_test.flatten() == predict.flatten())/ len(y_test)

    print(acc)
