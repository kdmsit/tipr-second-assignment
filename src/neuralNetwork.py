import numpy as np
import random

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
    #return (x) * (1.0 - (x))

class NeuralNetwork:
    def initialize_parameters(X, h, out):
        input_dim = len(X[0])
        w1 = np.random.rand(input_dim, h)
        w2 = np.random.rand(h, out)
        return w1, w2

    def forward_prop(X, w1, w2):
        z1 = np.dot(X, w1)
        y1 = sigmoid(z1)
        zout = np.dot(y1, w2)
        yout = sigmoid(zout)
        return w1, w2, z1, zout, y1, yout

    def back_prop(X, Y, w1, w2, z1, zout, y1, yout, learning_rate):
        # delta3 = 2 * (Y - output) * sigmoid_derivative(output)
        delta3 = 2 * (Y - yout) * sigmoid_derivative(zout)
        d_weights2 = np.dot(y1.T, delta3)
        # delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
        delta2 = np.dot(delta3, w2.T) * sigmoid_derivative(z1)
        d_weights1 = np.dot(X.T, delta2)
        w1 += learning_rate * d_weights1
        w2 += learning_rate * d_weights2
        return w1, w2


if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    learning_rate=0.3
    w1,w2 = NeuralNetwork.initialize_parameters(X, 2, 1)
    for i in range(6000):
        lw1, w2, z1, zout, y1, yout=NeuralNetwork.forward_prop(X, w1,w2)
        w1, w2=NeuralNetwork.back_prop(X, Y, w1, w2, z1, zout, y1, yout, learning_rate)
    accuracy=0
    prediction = NeuralNetwork.forward_prop(X, w1, w2)[5]
    for i in range(len(prediction)):
        if(prediction[i]>0.8):
            if(Y[i]==1.0):
                accuracy=accuracy+1
        else:
            if (Y[i] == 0.0):
                accuracy = accuracy + 1
    print(accuracy/4)