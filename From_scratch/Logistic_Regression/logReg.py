import numpy as np
import matplotlib.pyplot as plt
from from_scratch import LogisticRegression
from sklearn import datasets

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

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


x_train, x_test, y_train, y_test = train_test_split(x, y, split = 0.3, random=42)

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train_scaled = (x_train - mean)/std
x_test_scaled = (x_test - mean)/std

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

predict = model.predict(x_test_scaled)

def metrics(y_true, pred):
    
    y_true = np.array(y_true)
    pred = np.array(pred)
    
    tp = np.sum((y_true == 1) & (pred ==1))
    tn = np.sum((y_true == 0) & (pred ==0))
    fp = np.sum((y_true == 0) & (pred ==1))
    fn = np.sum((y_true == 1) & (pred ==0))
    
    accuracy = (tp+tn)/len(y_true)
    recall = tp/(tp+fn + 1e-15)
    precision = tp/(tp+fp + 1e-15)
    
    return accuracy, recall, precision

accuracy, recall, precision = metrics(y_test, predict)

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall:   {recall:.2f}")
print(f"Precision: {precision:.2f}")



from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predict)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()