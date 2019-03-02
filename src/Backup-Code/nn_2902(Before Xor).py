# Implement Neural Network here!
import numpy as np

# region Sigmoid Activation Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))
# endregion

# region Derivatives of Sigmoid Function
def diffSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
# endregion

def initialize_parameters(X,h1,h2,output_dim=10):
    model = {}
    input_dim = len(X[0])
    model['W1'] = np.random.randn(input_dim, h1) / np.sqrt(input_dim)
    model['b1'] = np.zeros((1, h1))
    model['W2'] = np.random.randn(h1, h2) / np.sqrt(h1)
    model['b2'] = np.zeros((1, h2))
    model['W3'] = np.random.randn(h2, output_dim) / np.sqrt(h2)
    model['b3'] = np.zeros((1, output_dim))
    return model

def forward_prop(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    z1 = np.dot(x,W1)+ b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1,W2)+ b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2,W3)+ b3
    exp_scores = np.exp(z3)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1, a1, z2, a2, z3, out


def back_prop(X,y,model,z1,a1,z2,a2,z3,output,reg_lambda):
    delta3 = output-y
    #delta3[range(len(X[0])), y] -= 1
    dW3 = (a2.T).dot(delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(model['W3'].T) * diffSigmoid(a2)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0)
    delta1 = delta2.dot(model['W2'].T) * diffSigmoid(a1)
    dW1 = np.dot(np.transpose(X), delta1)
    db1 = np.sum(delta1, axis=0)
    # Add regularization terms
    dW3 += reg_lambda * model['W3']
    dW2 += reg_lambda * model['W2']
    dW1 += reg_lambda * model['W1']
    return dW1, dW2, dW3, db1, db2, db3



def compute_cost(model,X,y,reg_lambda):
    # Code for cost function
    num_examples = len(X)
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation to calculate our predictions
    z1, a1, z2, a2, z3, out = forward_prop(model, X)
    probs = out / np.sum(out, axis=1, keepdims=True)
    probslist=list(probs)
    # Calculating the loss
    corect_logprobs = -np.log(probslist[range(num_examples), y])
    loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    return 1. / num_examples * loss


def train(model, X, y,epoc=1000, reg_lambda = .1, learning_rate=0.1):
    # Train your model
    # Batch gradient descent
    done = False
    previous_loss = float('inf')
    i = 0
    losses = []
    #while done == False:  # comment out while performance testing
    while i < epoc:
        # feed forward
        print(i)
        z1, a1, z2, a2, z3, output = forward_prop(model, X)
        # backpropagation
        dW1, dW2, dW3, db1, db2, db3 = back_prop(X, y, model, z1, a1, z2, a2, z3, output, reg_lambda)
        # update weights and biases
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        model['W3'] -= learning_rate * dW3
        model['b3'] -= learning_rate * db3
        '''if i % 1000 == 0:
            loss = compute_cost(model, X, y,reg_lambda)
            losses.append(loss)
            print("Loss after iteration %i: %f" % (i, loss))  # uncomment once testing finished, return mod val to 1000
            if (previous_loss - loss) / previous_loss < 0.01:
                done = True
                # print i
            previous_loss = loss'''
        i += 1
    return model




#def predict():
	# Code for prediction
