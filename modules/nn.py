import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def identity(x):
    return x

def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - max_x)
    return exp/exp.sum(axis=-1, keepdims=True)

class LinearLayer:
    
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = np.array(b)

    def forward(self, x):
        return np.dot(x, self.w) + self.b

def MSE(y, t):
    return 0.5 * np.sum((y-t)**2)

def CrossEntropyOneHotEncoded(y, t):
    if y.ndim == 1:
        y, t = y.reshape(1,-1), t.reshape(1, -1)
    
    delta = 1e-7 # log가 무한으로 발산하지 않게 하기 위한
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+delta)) / batch_size

def CrossEntropy(y, t):
    if y.ndim == 1:
        y, t = y.reshape(1,-1), t.reshape(1, -1)
    
    delta = 1e-7 # log가 무한으로 발산하지 않게 하기 위한
    batch_size = y.shape[0]
    return -np.sum((np.eye(y.shape[-1])[t])*np.log(y+delta)) / batch_size