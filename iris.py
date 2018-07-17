## IRIS

import numpy as np
from dnn.dnn import DNN

## Converting label name to onehot encoding
def label_encode(label):
	val=[]
	if label == "Iris-setosa":
		val = [1,0,0]
	elif label == "Iris-versicolor":
		val = [0,1,0]
	elif label == "Iris-virginica":
		val = [0,0,1]
	return val

## Converting data from the file to the appropriate format
def data_encode(file):
	X = []
	Y = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		X.append([line[0], line[1], line[2], line[3]])
		Y.append(label_encode(line[4]))
	return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

#Training
train_X , train_Y = data_encode('dataset/iris.train')

input_shape = train_X.shape
output_shape = train_Y.shape[-1]

## Initializing the DNN Class
nn = DNN(shape=input_shape)

## Adding the layers to the network
## You need to provide the number of node and the activation function.
## Sigmoid is the default activation function.
## sigmoid and tanh are the only activation function available

#nn.add(8, activation='tanh')
#nn.add(16, activation='sigmoid')
#nn.add(32, activation='sigmoid')
#nn.add(64, activation='sigmoid')
#nn.add(output_shape, activation='sigmoid')

## Here:
## 		epochs: training iterations
##		display_step: the iteration interval at which the training information should be displayed
##		batch_size: the batch of the data you want to give to the network during training
##		lr: the learning rate
##		validation_split: it split the dataset into the validation dataset and the training dataset
##		metrics: currently it onl support 'accuracy', it you specify it, then it will provide you withn the
##				accuracy of the training data during the training

#nn.compile(epochs=1000, display_step=100, batch_size=64, lr=0.0005, validation_split=0.2, metrics='accuracy')

## Loading the saved model
## Once your have a save model file then you need not to specify the layers and need not to call the compile()
## function.
## The saved model have all the data regarding your network, including your layers, weights, activation functions,
## the input_shape, all the value you give in the compile function.
## if you want to train you model then you can call the compile function overwrite all those value that the
## saved model have specified.
## You cannot change the weights, activation function, layers, i.e., you cannot change anything regarding the
## layers of your network

nn.load_model(path='dataset/iris')

## Training the model
## You need to provide current format data to the network as no preprocessing of data will be done y the network
## nn.train(training_dataset_input, training_dataset_output)
#nn.train(train_X, train_Y)

## Saving the model
#nn.save_model(path='dataset/iris')

## Evaluating the model, it returns the accuracy that the model has achieved
print(nn.evaluate(train_X, train_Y))
