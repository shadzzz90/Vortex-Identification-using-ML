import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers

from sklearn.metrics import accuracy_score

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

nn_architecture = [{"input_dim":2 , "output_dim":25, "activation":"relu"},
                   {"input_dim":25 , "output_dim":50, "activation":"relu"},
                   {"input_dim":50 , "output_dim":50, "activation":"relu"},
                   {"input_dim":50 , "output_dim":25, "activation":"relu"},
                   {"input_dim":25 , "output_dim":1, "activation":"sigmoid"}]


def init_layers(nn_architecture, seed = 2):
    np.random.seed(seed)

    num_of_layers = len(nn_architecture)

    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx +1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer ["output_dim"]
        params_values['W'+str(layer_idx)] = np.random.randn(layer_output_size,layer_input_size)*0.1
        params_values['b'+str(layer_idx)] = np.random.randn(layer_output_size,1)*0.1
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def single_layer_forward_propogation(A_prev, W_curr, b_curr, activation='relu'):
    Z_curr = np.dot(W_curr,A_prev)+b_curr

    if activation is "relu":
        activation_fun =relu
    elif activation is "sigmoid":
        activation_fun = sigmoid

    else:
        raise Exception('Non-supported activation function')

    return activation_fun(Z_curr), Z_curr

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]

    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T)+np.dot(1-Y,np.log(1-Y_hat).T))

    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def full_forward_propagation(X, params_values, nn_architecture):

    memory ={}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx+1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values['W'+ str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propogation(A_prev, W_curr, b_curr,activ_function_curr)

        memory["A"+str(idx)] = A_prev
        memory["Z"+str(layer_idx)] = Z_curr

    return A_curr, memory


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)


    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr


    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx+1)] -= learning_rate * grads_values["dW" + str(layer_idx+1)]
        params_values["b" + str(layer_idx+1)] -= learning_rate * grads_values["db" + str(layer_idx+1)]

    return params_values


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)


    return params_values, cost_history, accuracy_history


X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# transpose_X_train = np.transpose(X_train)
#
# # print(transpose_X_train.shape)
# y_train2 = y_train.reshape(y_train.shape[0],1)
#
# transpose_y_train = np.transpose(y_train2)
#
# print(transpose_y_train, transpose_y_train.shape, transpose_X_train, transpose_X_train.shape)
params_values, cost_history, acc_history = train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), nn_architecture , 10000, 0.01)
epochs = range(10000)
plt.plot(epochs, acc_history)

plt.show()

# print(X,y)