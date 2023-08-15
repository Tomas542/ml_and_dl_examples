import numpy as np

def sigmond(x):
    return 1 / (1 + np.exp(-x))

class Neuron():
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def y(self, x):
        s = np.dot(x, self.w) + self.b
        return sigmond(s)

Xi = np.array([2, 3])
Wi = np.array([0, 1])
bias = np.arange(-4, 4, 1)
print("Single Neuron:")
for i in bias:
    n = Neuron(Wi, i)
    print("Y =", n.y(Xi))

class OurNeuralNetwork():
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feeddorward(self, x):
        out_h1 = self.h1.y(x)
        out_h2 = self.h2.y(x)
        out_o1 = self.o1.y(np.array([out_h1, out_h2]))

        return out_o1
    
network = OurNeuralNetwork()
X = np.array([2, 3])
print('\n', "Network:")
print("Y =", network.feeddorward(X))

def mse_lose(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()