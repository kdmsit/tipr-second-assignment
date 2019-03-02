# Implement Neural Network here!
import numpy as np
import random

# region Sigmoid Activation Function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
# endregion

# region Derivatives of Sigmoid Function
def diffSigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
# endregion

def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))

def initialize_parameters(X,h,output_dim):
    model = {}
    input_dim = len(X[0])
    model['W1'] = np.random.randn(input_dim, h)
    model['b1'] = np.zeros((1, h))
    model['W2'] = np.random.randn(h, output_dim)
    model['b2'] = np.zeros((1, output_dim))
    return model

def forward_prop(model, x):
    W1, b1, W2, b2, = model['W1'], model['b1'], model['W2'], model['b2']
    z2 = np.dot(x, W1) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W2) + b2
    exp_scores = np.exp(z3)
    output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z2, a2, z3,output

def back_prop(x, y, model, z2, a2, z3, output,reg_lambda):
    a1=np.transpose(x)
    #a1=a1.reshape((2,1))
    W1, b1, W2, b2, = model['W1'], model['b1'], model['W2'], model['b2']
    delta3 = 2*(y - output)*(diffSigmoid(z3))
    dW2 = (a2.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)

    delta2=np.dot(delta3, W2) * diffSigmoid(z2)
    dW1 = np.dot(a1,delta2)
    db1 = np.sum(delta2, axis=0)

    # Add regularization terms
    #dW2 += reg_lambda * model['W2']
    #dW1 += reg_lambda * model['W1']
    return  dW1, dW2, db1, db2

#def compute_cost(model,X,y,reg_lambda):

def train(model, X, Y,epoc,learning_rate,reg_lambda):
    i=0
    while (i<epoc):
        print(i)
        #k=random.randint(0,len(X)-1)
        #x=X[k]
        #y=Y[k]
        z2, a2, z3, output= forward_prop(model,X)
        dW1, dW2, db1, db2 = back_prop(X, Y, model, z2, a2, z3, output,reg_lambda)
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        i=i+1
    return model

#def predict():
	# Code for prediction
