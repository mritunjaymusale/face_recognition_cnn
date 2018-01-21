import numpy as np
import tflearn
import cv2
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

#load data from compressed array 
temp_zip = np.load('data.npz')
X_test = temp_zip['X_test']
X_train = temp_zip['X_train']
Y_test = temp_zip['Y_test']
Y_train = temp_zip['Y_train']
num_classes = Y_train.shape[1]

# Building convolutional network
network = input_data(shape=[None, 150, 150, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, num_classes, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy', name='target')

model =tflearn.DNN(network, tensorboard_verbose=0)
model.load('./model/model')


pred_values = model.predict(X_test)

for i in range(len(pred_values)):
    for j in range(i) :
        print('predicted values :',pred_values[i][j])
    print('actual vaules :',Y_test[i])
