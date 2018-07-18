# Feedforward Neural Network

It is a simple Feedforward Neural Network that is build from scratch using Numpy and Python3. I developed this project as a self interest
to learn the working of a neural network, to understand the backend this of a neural network.

## Getting Started

Just simply download this code and first run the iris.py file. The iris.py has the neural network that is trained on IRIS Datatset.
It will help you to build your own neural network.

### Prerequisites

You only need **numpy** installed on your python3 environment.
> sudo pip3 install numpy

### Example

First import the DNN class from the dnn directory and dnn.py file
```
from dnn.dnn import DNN
```

Create an object of the class
```
input_shape = (4, 2)
nn = DNN(shape=input_shape)
```

The input shape it the shape of the input of your training dataset.
Example: (4, 2)
4: The number of exmaples in the dataset, you can vary this during initialize, it does have any effect.
2: The size of the input data, this is really very important.

Now add layers to it
```
nn.add(5, activation='tanh')
nn.add(1, activation='sigmoid')
```

Currently this project supports only **sigmoid** and **tanh** activation function. Sigmoid is the default activation function.

Now compile your neural network

```
nn.compile(epochs=1000, display_step=100, batch_size=64, lr=0.05, validation_split=0.2, metrics='accuracy')
```

The compile function is used to give all the necessary information to the neural network. It is a compulsoty function, you need to call 
it before working on training.

* **epochs**: training iterations
* **display_step**: the iteration interval at which the training information should be displayed
* **batch_size**: the batch of the data you want to give to the network during training
* **lr**: the learning rate
* **validation_split**: it split the dataset into the validation dataset and the training dataset
* **metrics**: currently it only support **accuracy**, you specify it, then it will provide you with the accuracy of the training 
    data during the training
    
```
nn.load_model(path='dataset/iris')
```

**nn.load_model(path=None)** - This function is used for loading the saved model.

Once your have a save model file then you need not to specify the layers and need not to call the compile()
function.

The saved model have all the data regarding your network, including your layers, weights, activation functions,
the input_shape, all the value you give in the compile function.

if you want to train you model then you can call the compile function overwrite all those value that the
saved model have specified.

You cannot change the weights, activation function, layers, i.e., you cannot change anything regarding the
layers of your network.

```
nn.train(train_X, train_Y)
```

**nn.train(X, Y)** - This function is used to train the model.
You need to provide current format data to the network as no preprocessing of data will be done y the network

```
nn.save_model(path='dataset/iris')
```
Saving the model

```
print(nn.evaluate(train_X, train_Y))
```
Evaluating the model, it returns the accuracy that the model has achieved

### Complete Code
```
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

#Training Dataset
train_X , train_Y = data_encode('dataset/iris.train')

input_shape = train_X.shape
output_shape = train_Y.shape[-1]

## Initializing the DNN Class
nn = DNN(shape=input_shape)

nn.add(8, activation='tanh')
nn.add(16, activation='sigmoid')
nn.add(32, activation='sigmoid')
nn.add(64, activation='sigmoid')
nn.add(output_shape, activation='sigmoid')

nn.compile(epochs=1000, display_step=100, batch_size=64, lr=0.05, validation_split=0.2, metrics='accuracy')

nn.train(train_X, train_Y)
nn.save_model(path='dataset/iris')
print(nn.evaluate(train_X, train_Y))
```

## Some important functions of DNN class

* add(num_of_node, activation='sigmoid')
	> It will add layers in the neural network
* summary()
	> Give a summary of the neural network, like: layers, no of nodes, activation function used.
* compile(loss='absolute_error', batch_size=32, optimizer='sgd', epochs=1000,
	display_step=100, lr=0.01, validation_split=0.0, metrics=None)
	> It helps in providing the required information to the neural network
* save_weights(path=None)
	> Save the weights of the neural network
* load_weights(self, path=None)
	> Load the saved weights
* save_model(path=None)
	> Save the entire neural network model
* load_model(path=None)
	> Load the entire neural network model
* predict(X)
	> Get the result of input datset from the neural network
* train(X, Y)
	> Train the neural network on the datat provided.
* evaluate(X, Y)
	> It returns the accuracy of a dataset.
