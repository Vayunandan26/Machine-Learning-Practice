import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x, y = datasets.make_regression(n_samples=600, n_features=1, noise=20, random_state=10)

def train_test_split(x, y, split = 0.2, random = None):
    
    if random is None :
        np.random.seed(random)
    
    assert len(x) == len(y)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    split_count = int(len(x)*split)
    
    train_indices = indices[split_count:]
    test_indices = indices[:split_count]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices] 

def mean_sqaured_error(actual, y_predict):
    mean = np.mean((actual-y_predict)**2)
    return mean

x_train, x_test, y_train, y_test = train_test_split(x, y, split=0.3, random=42)

from from_scratch import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

predict = model.predict(x_test)

error = mean_sqaured_error(y_test, predict)

print(error)

y_pred_line = model.predict(x)
c_map = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,3))
m1 = plt.scatter(x_train, y_train, color=c_map(0.9), s=10)
m1 = plt.scatter(x_test, y_test, color=c_map(0.5), s=10)
plt.plot(x, y_pred_line, color='black', linewidth = 2, label = 'prediction')
plt.show()