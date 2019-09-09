import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

N_SAMPLES = 1000
TEST_SIZE =0.1

NNStructure = [{"input layer": 2, "output layer": 25, "activation":"relu"},
                {"input layer": 25, "output layer": 50, "activation":"relu"},
                {"input layer": 50, "output layer": 50, "activation":"relu"},
                {"input layer": 50, "output layer": 25, "activation":"relu"},
                {"input layer": 25, "output layer": 1, "activation":"sigmoid"}]

def initailizeNN (NNStructure):
    np.random.seed(99)
    parameter_values = {}
    for index, layer in enumerate(NNStructure):
        layer_index = index+1
        parameter_values["W"+str(layer_index)] = np.random.randn(layer["output layer"], layer["input layer"])*0.1
        parameter_values["b" + str(layer_index)] = np.random.randn(layer["output layer"], 1) * 0.1

    return parameter_values

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]

    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T)+np.dot(1-Y,np.log(1-Y_hat).T))

    return np.squeeze(cost)

def forwardPropagation (X, parameter_values, NNStructure):

    A_prev = X
    memory = {}
    memory["A"+str(0)] = A_prev
    for index, layer in enumerate(NNStructure):
        layer_index = index+1
        Z_curr = np.dot(parameter_values["W"+str(layer_index)], A_prev)+parameter_values["b"+str(layer_index)]

        if layer["activation"] is "relu":
            A_prev = relu(Z_curr)
        elif layer["activation"] is "sigmoid":
            A_prev = sigmoid(Z_curr)

        memory['Z' + str(layer_index)] = Z_curr
        memory["A"+str(layer_index)] = A_prev

    return A_prev, memory

#
def backwardPropagation(Y_hat, Y, memory, parameter_values, NNStructure):
    grad_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_index, layer in reversed(list(enumerate(NNStructure))):
        layer_current_index = layer_index + 1

        if layer['activation'] is "relu":
            dZ = relu_backward(dA_prev, memory["Z" + str(layer_current_index)])

        elif layer['activation'] is "sigmoid" :
            dZ = sigmoid_backward(dA_prev, memory["Z" + str(layer_current_index)])

        # temp=np.transpose(memory["A"+str(layer_index)])
        dW = np.dot(dZ, np.transpose(memory["A"+str(layer_index)]))/m
        db = np.sum(dZ, axis=1 , keepdims=True)/m
        dA = np.dot(parameter_values["W"+str(layer_current_index)].T, dZ)

        dA_prev = dA

        grad_values["dW"+str(layer_current_index)] = dW
        grad_values["db"+str(layer_current_index)] = db

    return grad_values


def update(grad_values, parameter_values, NNStructure, learningrate):

    for layer_index, layer in enumerate(NNStructure):
        parameter_values["W"+str(layer_index+1)] -= grad_values["dW"+str(layer_index+1)]*learningrate
        parameter_values["b" + str(layer_index+1)] -= grad_values["db" + str(layer_index+1)] * learningrate

    return parameter_values



def trainNN(X, Y, NNStructure, epochs, learningrate):

    parameter_values = initailizeNN(NNStructure)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, memory = forwardPropagation(X=X, parameter_values=parameter_values, NNStructure= NNStructure)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        grad_values = backwardPropagation(Y_hat=Y_hat, Y=Y, memory=memory, parameter_values= parameter_values, NNStructure=NNStructure)
        parameter_values = update(grad_values=grad_values, parameter_values=parameter_values, NNStructure=NNStructure, learningrate=learningrate)

    return parameter_values, cost_history, accuracy_history

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)


# print(np.transpose(Y.reshape(Y.shape[0],1)))

last_param, cost_history, accuracy_history = trainNN(np.transpose(X_train),np.transpose(y_train.reshape(y_train.shape[0],1)),
                                                     NNStructure, 10000, learningrate=0.01)

epochs = range(10000)
plt.plot(epochs, accuracy_history)

plt.show()
# print(forwardPropagation(np.transpose(X), parameter_values))

# print(list(enumerate(parameter_values)))

# for layer_index, layer in reversed(list(enumerate(NNStructure))):
#     print(layer_index, layer)