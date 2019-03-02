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

def initialize_parameters(X,h,out):
    model = {}
    input_dim = len(X[0])
    model['weights1'] = np.random.rand(input_dim, h)
    model['weights2'] = np.random.rand(h,out)
    return model

def forward_prop(X,model):
    weights1, weights2 = model['weights1'], model['weights2']
    layer1 = sigmoid(np.dot(X, weights1))
    output = sigmoid(np.dot(layer1, weights2))
    model['output'] = output
    return layer1, output, model

def back_prop(X,Y,model,layer1,learning_rate):
    weights1, weights2, output = model['weights1'], model['weights2'], model['output']
    delta3 = 2 * (Y - output) * sigmoid_derivative(output)
    d_weights2 = np.dot(layer1.T, delta3)
    delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
    d_weights1 = np.dot(X.T, delta2)
    model['weights1'] += learning_rate * d_weights1
    model['weights2'] += learning_rate * d_weights2
    return model

#def compute_cost(model,X,y,reg_lambda):

def train(model, X, Y,epoc,learning_rate):
    thismodel=model
    for i in range(epoc):
        layer1, output, thismodel = forward_prop(X, model)
        thismodel=back_prop(X, Y, thismodel, layer1, learning_rate)
    return thismodel

def predict(X,Y,model):
    # Code for prediction
    accuracy = 0
    prediction = forward_prop(X, model)[1]
    exp_scores = np.exp(prediction)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    for i in range(len(out)):
        pred=list(out[i])
        index=pred.index(max(pred))
        #print(index,Y[i])
        if(index==Y[i]):
            accuracy=accuracy+1
    return accuracy /len(Y)
