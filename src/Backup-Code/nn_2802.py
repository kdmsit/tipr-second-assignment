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

def initialize_parameters(imagePixelList):
    # Code for parameter initialization
    np.random.seed(0)
    epoch = 5000
    # neural network hyperparameters
    no_of_layers = 4
    input_layer_size = len(imagePixelList[1])
    h1_size = 1500
    h2_size = 100
    output_layer_size = 10
    lmbda = 2
    # weight and bias initialization
    w1 = np.random.rand(input_layer_size,h1_size)
    b1 = np.random.rand(1,h1_size)
    w2 = np.random.rand(h1_size, h2_size)
    b2 = np.random.rand(1, h2_size)
    w3 = np.random.rand(h2_size, output_layer_size)
    b3 = np.random.rand(1,output_layer_size)
    return input_layer_size, h1_size,h2_size, output_layer_size, lmbda,w1,b1,w2,b2,w3,b3,epoch

def forward_prop(a1,w1,b1,w2,b2,w3,b3):
    # Code for forward propagation
    #z2 = np.dot(np.transpose(a1), w1) + b1
    z2 = np.dot(np.transpose(a1), w1)
    a2 = sigmoid(z2)
    #z3 = np.dot(a2, w2) + b2
    z3 = np.dot(a2, w2)
    a3 = sigmoid(z3)
    #z4 = np.dot(a3, w3) + b3
    z4 = np.dot(a3, w3)
    a4 = sigmoid(z4)
    return  z2,a2,z3,a3,z4,a4

def back_prop(lmbda,a1,w1,b1,a2,z2,w2,b2,a3,z3,w3,b3,z4,a4,yactual):
    # Code for backward propagation
    delta4 =(yactual - a4) * diffSigmoid(a4)
    d_weights3 = lmbda * np.dot(np.transpose(a3), delta4)
    #db2 = np.sum(delta3, axis=0, keepdims=True)

    delta3 = np.dot(delta4, np.transpose(w3)) * diffSigmoid(a3)
    d_weights2 = lmbda * np.dot(np.transpose(a2), delta3)
    #db1 = np.sum(delta2, axis=0)

    delta2 = np.dot(delta3, np.transpose(w2)) * diffSigmoid(a2)
    d_weights1 = lmbda * np.dot(a1, delta2)
    #db1 = np.sum(delta2, axis=0)

    w1 += - d_weights1
    #b1 += - db1
    w2 += - d_weights2
    #b2 += - db2
    w3 += - d_weights3
    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2, 'W3': w3, 'b3': b3}
    return model



#def compute_cost():
	# Code for cost function


#def train():
	# Train your model


#def predict():
	# Code for prediction
