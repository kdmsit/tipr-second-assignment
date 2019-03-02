# Implement Neural Network here!
import numpy as np
import random

# region Sigmoid Activation Function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
# endregion

# region Derivatives of Sigmoid Function
def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
# endregion

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_prime(z):
    return softmax(z) * (1 - softmax(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - tanh(z) * tanh(z)


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return 1. * (z > 0)


def initialize_parameters(layer):
    weights = []
    for i in range(len(layer) - 1):
        j = i + 1
        weight = np.random.rand(layer[i], layer[j])
        weights.append(weight)
    return weights


def forward_prop(X, weights):
    layerout = []
    Z = []
    y = X
    layerout.append(y)
    Z.append([])
    for w in weights:
        z = np.dot(y, w)
        #y = sigmoid(z)
        y = relu(z)
        layerout.append(y)
        Z.append(z)
    return layerout, Z


def back_prop(layerout,Y, Z, weights, learning_rate):
    noofLayers = len(layerout)
    delta = []
    dweights = []
    yout = layerout[noofLayers - 1]
    zout = Z[noofLayers - 1]
    # delta.append(deltaout)
    i = noofLayers - 1
    while (i > 0):
        if (i == noofLayers - 1):
            yout = layerout[i]
            zout = Z[i]
            #thisdelta = (Y - yout) * sigmoid_derivative(zout)
            thisdelta = (Y - yout) * relu_prime(zout)
        else:
            w = weights[i]
            z = Z[i]
            #thisdelta = np.dot(delta, w.T) * sigmoid_derivative(z)
            thisdelta = np.dot(delta, w.T) * relu_prime(z)
        delta = thisdelta
        y = layerout[i - 1]
        dweight = np.dot(y.T, thisdelta)
        dweights.append(dweight)
        i = i - 1
    dweights.reverse()
    for i in range(len(weights)):
        weights[i] += learning_rate * dweights[i]
    return weights

#def compute_cost(model,X,y,reg_lambda):

def train(X,Y, weights,learning_rate):
    layerout, Z = forward_prop(X, weights)
    weights = back_prop(layerout, Y,Z, weights, learning_rate)
    return weights

def predict(X,Y,weights):
    # Code for prediction
    accuracy = 0
    out = forward_prop(X, weights)[0]
    k = len(out)
    prediction = out[k - 1]
    '''for i in range(len(prediction)):
        if (prediction[i] > 0.8):
            if (Y[i] == 1.0):
                accuracy = accuracy + 1
        else:
            if (Y[i] == 0.0):
                accuracy = accuracy + 1
    print(accuracy / 4)'''

    exp_scores = np.exp(prediction)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    for i in range(len(out)):
        pred=list(out[i])
        index=pred.index(max(pred))
        #print(index,Y[i])
        if(index==Y[i]):
            accuracy=accuracy+1
    return accuracy /len(Y)
