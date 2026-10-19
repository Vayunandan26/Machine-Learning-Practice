import numpy as np 

class NB :
    def fit (self, x, y):
        n_samples, n_features = x.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes , dtype=np.float64)
        
        for i in self.classes:
            xi = x[i==y]
            self.mean[i, :] = xi.mean(axis=0)
            self.var[i, :] = xi.var(axis=0)   
            self.priors[i] = xi.shape[0] / float(n_samples)
            
    def probdense(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        
        num = np.exp((-(x-mean)**2 )/ (2 * (var)))
        den = np.sqrt(2 * np.pi * var)
        
        return num / den 
    
    def _predict (self, x):

        poster = []
        
        for idx, _ in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            cond = np.sum(np.log(self.probdense(idx, x)))
            
            posters = prior + cond 
            poster.append(posters)
            
        return self.classes[np.argmax(poster)]
    
    def predict(self, x):
        return [self._predict(i) for i in x ] 
        
        