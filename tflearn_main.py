import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
model_location='./model/'


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
# 
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit({'input':X_train},{'target':Y_train},n_epoch=65,validation_set=({'input':X_test},{'target':Y_test}),
                            snapshot_epoch=100,show_metric=True)


# model.save(model_location)