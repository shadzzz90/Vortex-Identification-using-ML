import numpy as np
from sklearn.datasets import make_moons

N_SAMPLES = 10

NNStructure = [{"input layer": 2, "output layer": 4, "activation":"relu"},
                {"input layer": 4, "output layer": 6, "activation":"relu"},
                {"input layer": 6, "output layer": 8, "activation":"relu"},
                {"input layer": 8, "output layer": 1, "activation":"sigmoid"}]

def initailizeNN (NNStructure):
    np.random.seed(99)
    parameter_values = {}
    for index, layer in enumerate(NNStructure):
        layer_index = index+1
        parameter_values["W"+str(layer_index)] = np.random.randn(layer["output layer"], layer["input layer"])*0.1
        parameter_values["b" + str(layer_index)] = np.random.randn(layer["output layer"], 1) * 0.1

    return parameter_values

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

    return memory

#
def backwardPropagation(Y_hat, Y, memory, parameter_values,  NNStructure):
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

        dW = np.dot(dZ, parameter_values["A"+str(layer_index)].T)
        db = np.sum(dZ, axis=1 , keepdims=True)/m
        dA = np.dot(parameter_values["W"+str(layer_current_index)].T, dZ)

        dA_prev = dA

        grad_values["dW"+str(layer_current_index)] = dW
        grad_values["db"+str(layer_current_index)] = db

    return grad_values




X, Y = make_moons(n_samples=N_SAMPLES, shuffle=True, noise=True, random_state=20)


parameter_values = initailizeNN(NNStructure)

#
# print(parameter_values)
# print(forwardPropagation(np.transpose(X), parameter_values))

# print(list(enumerate(parameter_values)))

for layer_index, layer in reversed(list(enumerate(NNStructure))):
    print(layer_index, layer)