import numpy as np
import random

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
    #return (x) * (1.0 - (x))

class NeuralNetwork:
    def initialize_parameters(X, h1, h2, out):
        model = {}
        input_dim = len(X[0])
        model['weights1'] = np.random.rand(input_dim, h1)
        model['weights2'] = np.random.rand(h1, h2)  # layer-2
        model['weights3'] = np.random.rand(h2, out)
        return model

    def forward_prop(X, model):
        weights1, weights2, weights3 = model['weights1'], model['weights2'], model['weights3']
        ita1 = np.dot(X, weights1)
        layer1 = sigmoid(ita1)
        ita2 = np.dot(layer1, weights2)
        layer2 = sigmoid(ita2)
        itaout = np.dot(layer2, weights3)
        output = sigmoid(itaout)
        model['output'] = output
        model['ita1'] = ita1
        model['ita2'] = ita2
        model['itaout'] = itaout
        return layer1, layer2, output, model

    def back_prop(X, Y, model, layer1, layer2, learning_rate):
        weights1, weights2, weights3, output, ita1, ita2, itaout = model['weights1'], model['weights2'], model[
            'weights3'], model['output'], model['ita1'], model['ita2'], model['itaout']
        # delta4 = (Y - output) * sigmoid_derivative(output)
        delta4 = (Y - output) * sigmoid_derivative(itaout)
        d_weights3 = np.dot(layer2.T, delta4)
        # delta3 = np.dot(delta4, weights3.T) * sigmoid_derivative(layer2)
        delta3 = np.dot(delta4, weights3.T) * sigmoid_derivative(ita2)
        d_weights2 = np.dot(layer1.T, delta3)
        # delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
        delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(ita1)
        d_weights1 = np.dot(X.T, delta2)
        model['weights1'] += learning_rate * d_weights1
        model['weights2'] += learning_rate * d_weights2
        model['weights3'] += learning_rate * d_weights3
        return model


if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    learning_rate=0.3
    model = NeuralNetwork.initialize_parameters(X, 2, 2, 1)
    for i in range(7000):
        layer1,layer2,output, model=NeuralNetwork.forward_prop(X,model)
        model=NeuralNetwork.back_prop(X, Y, model, layer1, layer2, learning_rate)
    accuracy=0
    prediction = NeuralNetwork.forward_prop(X, model)[2]
    for i in range(len(prediction)):
        if(prediction[i]>0.8):
            if(Y[i]==1.0):
                accuracy=accuracy+1
        else:
            if (Y[i] == 0.0):
                accuracy = accuracy + 1
    print(accuracy/4)