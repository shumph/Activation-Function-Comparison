import numpy as np
from keras.datasets import mnist

#set up our math functions
def sigmoid(value):
    sigmoid_value = 1/(1+np.exp(-value))
    return sigmoid_value

def ddx_sigmoid(value):
    ddx_sigmoid_value = sigmoid(value) * (1-sigmoid(value))    #s'(x) = s(x) * (1-s(x))
    return ddx_sigmoid_value

def softmax(value):
    value -= np.max(value, axis=0) 
    softmax_value = np.exp(value) / np.sum(np.exp(value), axis=0)
    return softmax_value

#functions for accessing current inputs, predictions, and accuracy
def get_predictions(active_layer_two):
    return np.argmax(active_layer_two, 0)

def get_accuracy(predictions, target):
    return np.sum(predictions == target)/target.size

def create_prediction(input, weight_one ,bias_one, weight_two, bias_two):
    x, y, z, active_layer_two = forward_propagation(input, weight_one, bias_one, weight_two, bias_two)
    predictions = get_predictions(active_layer_two)
    return predictions

def get_current_image(i ,input, weight_one, bias_one, weight_two, bias_two, w, h):
    vector_input = input[:, i,None]
    prediction = create_prediction(vector_input, weight_one, bias_one, weight_two, bias_two)
    current_image = vector_input.reshape((w, h)) * 255
    return (prediction, current_image)

#the good stuff
def construct_parameters(size):
    weight_one = np.random.rand(10,size) - 0.5
    bias_one = np.random.rand(10,1) - 0.5
    weight_two = np.random.rand(10,10) - 0.5
    bias_two = np.random.rand(10,1) - 0.5
    return weight_one, bias_one, weight_two, bias_two

def forward_propagation(input,weight_one,bias_one,weight_two,bias_two):
    layer_one = weight_one.dot(input) + bias_one 
    active_layer_one = sigmoid(layer_one) 
    layer_two = weight_two.dot(active_layer_one) + bias_two 
    active_layer_two = softmax(layer_two) 
    return layer_one, active_layer_one, layer_two, active_layer_two

def encoder(target):
    encoded_target = np.zeros((target.max()+1,target.size)) 
    encoded_target[target,np.arange(target.size)] = 1 
    return encoded_target

def backward_propagation(input, target, active_layer_one, active_layer_two, weight_two, layer_one, rows):
    encoded_target = encoder(target)
    ddx_layer_two = 2*(active_layer_two - encoded_target) 
    ddx_weight_two = 1/rows * (ddx_layer_two.dot(active_layer_one.T)) 
    ddx_bias_two = 1/rows * np.sum(ddx_layer_two,1) 
    ddx_layer_one = weight_two.T.dot(ddx_layer_two)*ddx_sigmoid(layer_one) 
    ddx_weight_one = 1/rows * (ddx_layer_one.dot(input.T)) 
    ddx_bias_one = 1/rows * np.sum(ddx_layer_one,1) 
    return ddx_weight_one, ddx_bias_one, ddx_weight_two, ddx_bias_two

def update_params(hyperparamater, weight_one, bias_one, weight_two, bias_two, ddx_weight_one, ddx_bias_one, ddx_weight_two, ddx_bias_two):
    weight_one -= hyperparamater * ddx_weight_one
    bias_one -= hyperparamater * np.reshape(ddx_bias_one, (10,1))
    weight_two -= hyperparamater * ddx_weight_two
    bias_two -= hyperparamater * np.reshape(ddx_bias_two, (10,1))
    return weight_one, bias_one, weight_two, bias_two

def gradient_descent_sigmoid(input, target, hyperparamater, iterations, w, h):
    size , rows = input.shape
    weight_one, bias_one, weight_two, bias_two = construct_parameters(size)
    accuracy_list = []
    image_list = []
    for i in range(iterations):
        layer_one, active_layer_one, layer_two, active_layer_two = forward_propagation(input, weight_one, bias_one, weight_two, bias_two)
        ddx_weight_one, ddx_bias_one, ddx_weight_two, ddx_bias_two = backward_propagation(input, target, active_layer_one, active_layer_two, weight_two, layer_one, rows)
        weight_one, bias_one, weight_two, bias_two = update_params(hyperparamater, weight_one, bias_one, weight_two, bias_two, ddx_weight_one, ddx_bias_one, ddx_weight_two, ddx_bias_two)   
        prediction = get_predictions(active_layer_two)
        accuracy_list.append(get_accuracy(prediction, target))
        image_list.append(get_current_image(i, input, weight_one, bias_one, weight_two, bias_two, w, h))
    return weight_one, bias_one, weight_two, bias_two, accuracy_list, image_list

def main():
    (input_train, target_train), (dummy, dummy) = mnist.load_data()
    w = input_train.shape[1]
    h = input_train.shape[2]
    input_train = input_train.reshape(input_train.shape[0],w*h).T / 255
    weight_one, bias_one, weight_two, bias_two, accuracy_list, image_list = gradient_descent_sigmoid(input_train, target_train, 0.15, 200, w, h)
    print(accuracy_list)
    #print(image_list)
#main()