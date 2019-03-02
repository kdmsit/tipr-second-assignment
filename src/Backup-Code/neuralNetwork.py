import numpy as np
import random

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

class NeuralNetwork:
    def initialise_param(X):
        model = {}
        input_dim = len(X[0])
        model['weights1'] = np.random.rand(input_dim, 4)
        model['weights2'] = np.random.rand(4,1)
        return model

    def feedforward(X,model):
        weights1,weights2=model['weights1'],model['weights2']
        layer1 = sigmoid(np.dot(X, weights1))
        #layer1=layer1.reshape(1,layer1.size)
        output = sigmoid(np.dot(layer1, weights2))
        model['output'] = output
        return layer1,output,model

    def backprop(x,y,model,layer1,learning_rate):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        weights1, weights2, output = model['weights1'], model['weights2'], model['output']
        delta3=2*(y - output) * sigmoid_derivative(output)
        d_weights2 = np.dot(layer1.T,delta3)
        delta2=np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
        d_weights1 = np.dot(x.T,delta2)
        # update the weights with the derivative (slope) of the loss function
        model['weights1'] += learning_rate*d_weights1
        model['weights2'] += learning_rate*d_weights2


if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])
    learning_rate=0.3
    model = NeuralNetwork.initialise_param(X)
    for i in range(5000):
        layer1, output, model=NeuralNetwork.feedforward(X,model)
        NeuralNetwork.backprop(X,Y,model,layer1,learning_rate)
    accuracy=0
    prediction = NeuralNetwork.feedforward(X, model)[1]
    for i in range(len(prediction)):
        if(prediction[i]>0.8):
            if(Y[i]==1.0):
                accuracy=accuracy+1
        else:
            if (Y[i] == 0.0):
                accuracy = accuracy + 1
    print(accuracy/4)