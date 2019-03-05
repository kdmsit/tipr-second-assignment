# Implement Neural Network here!
import numpy as np
from sklearn.metrics import f1_score
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

def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    return 1. * (z > 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_derivative(z):
    return softmax(z) * (1 - softmax(z))


def initialize_parameters(layer):
    weights = []
    for i in range(len(layer) - 1):
        j = i + 1
        weight = np.zeros(layer[i], layer[j])
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
        y = sigmoid(z)
        layerout.append(y)
        Z.append(z)
    return layerout, Z


def back_prop(layerout, Z,Y, weights, learning_rate):
    noofLayers = len(layerout)
    delta = []
    dweights = []
    i = noofLayers - 1
    while (i > 0):
        if (i == noofLayers - 1):
            yout = layerout[i]
            zout = Z[i]
            thisdelta = (Y - yout) * sigmoid_derivative(yout)
        else:
            w = weights[i]
            z = Z[i]
            y=layerout[i]
            thisdelta = np.dot(delta, w.T) * sigmoid_derivative(y)
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

def train(model, X, Y,weights,learning_rate):
    layerout, Z = forward_prop(X,weights)
    model = back_prop(layerout, Z,Y, weights, learning_rate)
    return model

def predict(X,Y,weights):
    # Code for prediction
    accuracy = 0
    predictions=[]
    out = forward_prop(X, weights)[0]
    k = len(out)
    prediction = out[k - 1]
    exp_scores = np.exp(prediction)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    for i in range(len(out)):
        pred=list(out[i])
        index=pred.index(max(pred))
        predictions.append(index)
        if(index==Y[i]):
            accuracy=accuracy+1
    accuracyOfMyCode = (accuracy / len(Y)) * 100.0
    f1_score_macro = f1_score(Y, predictions, average='macro')
    f1_score_micro = f1_score(Y, predictions, average='micro')
    return accuracyOfMyCode, f1_score_macro, f1_score_micro
