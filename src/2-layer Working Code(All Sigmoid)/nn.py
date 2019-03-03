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

def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    return 1. * (z > 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_derivative(z):
    return softmax(z) * (1 - softmax(z))

def initialize_parameters(X,h1,h2,out):
    model = {}
    input_dim = len(X[0])
    model['weights1'] = np.random.rand(input_dim, h1)
    model['weights2'] = np.random.rand(h1, h2)
    model['weights3'] = np.random.rand(h2,out)
    return model

def forward_prop(X,model):
    weights1, weights2,weights3 = model['weights1'], model['weights2'],model['weights3']
    layer1 = sigmoid(np.dot(X, weights1))
    #layer1 = tanh(np.dot(X, weights1))
    #layer1 = relu(np.dot(X, weights1))
    layer2 = sigmoid(np.dot(layer1, weights2))
    #layer2 = tanh(np.dot(layer1, weights2))
    #layer2 = relu(np.dot(layer1, weights2))
    output = sigmoid(np.dot(layer2, weights3))
    #output = softmax(np.dot(layer2, weights3))
    model['output'] = output
    return output,layer2,layer1, model

def back_prop(X,Y,model,layer1,layer2,learning_rate):
    weights1, weights2,weights3,output = model['weights1'], model['weights2'],model['weights3'], model['output']
    delta4 = 2 * (Y - output) * sigmoid_derivative(output)
    #delta4 = 2 * (Y - output) * softmax_derivative(output)
    d_weights3 = np.dot(layer2.T, delta4)
    delta3 = np.dot(delta4, weights3.T) * sigmoid_derivative(layer2)
    #delta3 = np.dot(delta4, weights3.T) * tanh_derivative(layer2)
    #delta3 = np.dot(delta4, weights3.T) * relu_derivative(layer2)
    d_weights2 = np.dot(layer1.T, delta3)
    delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
    #delta2 = np.dot(delta3, weights2.T) * tanh_derivative(layer1)
    #delta2 = np.dot(delta3, weights2.T) * relu_derivative(layer1)
    d_weights1 = np.dot(X.T, delta2)
    model['weights3'] += learning_rate * d_weights3
    model['weights1'] += learning_rate * d_weights1
    model['weights2'] += learning_rate * d_weights2
    return model

#def compute_cost(model,X,y,reg_lambda):

def train(model, X, Y,learning_rate):
    output,layer2,layer1, model = forward_prop(X, model)
    model = back_prop(X,Y,model,layer1,layer2,learning_rate)
    return model

def predict(X,Y,model):
    # Code for prediction
    accuracy = 0
    prediction = forward_prop(X, model)[0]
    exp_scores = np.exp(prediction)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    for i in range(len(out)):
        pred=list(out[i])
        index=pred.index(max(pred))
        #print(index,Y[i])
        if(index==Y[i]):
            accuracy=accuracy+1
    return accuracy /len(Y)
