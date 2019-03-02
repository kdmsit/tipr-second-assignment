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
    weight={}
    input_dim = len(X[0])
    weight[1] = np.random.rand(input_dim, h)
    weight[2] = np.random.rand(h,out)
    model['W']=weight
    return model

def forward_prop(X,model):
    ita={}
    layerout={}
    weight=model['W']
    weights1, weights2 = weight[1], weight[2]
    ita1=np.dot(X, weights1)
    layer1 = sigmoid(ita1)
    itaout=np.dot(layer1, weights2)
    output = sigmoid(itaout)
    layerout[2] = output
    layerout[1] = layer1
    ita[2] = itaout
    ita[1] = ita1
    model['ita'] = ita
    model['layerout'] = layerout
    return model

def back_prop(X,Y,model,learning_rate):
    weight=model['W']
    ita=model['ita']
    layerout=model['layerout']
    weights1, weights2= weight[1], weight[2]
    layer1,output=layerout[1],layerout[2]
    ita1, itaout=ita[1],ita[2]
    #delta3 = 2 * (Y - output) * sigmoid_derivative(output)
    delta3 = 2 * (Y - output) * sigmoid_derivative(itaout)
    d_weights2 = np.dot(layer1.T, delta3)
    #delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(layer1)
    delta2 = np.dot(delta3, weights2.T) * sigmoid_derivative(ita1)
    d_weights1 = np.dot(X.T, delta2)
    weight[1] += learning_rate * d_weights1
    weight[2] += learning_rate * d_weights2
    model['W'] = weight
    return model

#def compute_cost(model,X,y,reg_lambda):

def train(model, X, Y,learning_rate):
    model = forward_prop(X, model)
    model=back_prop(X, Y, model, learning_rate)
    return model

def predict(X,Y,model):
    # Code for prediction
    accuracy = 0
    model = forward_prop(X, model)
    layerout=model['layerout']
    prediction=layerout[2]
    exp_scores = np.exp(prediction)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    for i in range(len(out)):
        pred=list(out[i])
        index=pred.index(max(pred))
        #print(index,Y[i])
        if(index==Y[i]):
            accuracy=accuracy+1
    return accuracy /len(Y)
