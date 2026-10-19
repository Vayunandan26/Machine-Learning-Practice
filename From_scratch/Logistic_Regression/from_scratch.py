import numpy as np

class LogisticRegression :
    def __init__(self, lr = 0.01, n_iters = 30000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 
        self.loss_history = []
        
    def fit(self, x, y):
        
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear = np.dot(x, self.weights) + self.bias 
            y_pred = self.sigmoid(linear)
            
            epsilon = 1e-15
            loss = -(1/n_samples) * np.sum(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
            self.loss_history.append(loss)
            
            dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        
        linear = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        
        return [1 if i > 0.5 else 0 for i in y_pred]
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1/(1 + np.exp(-x))