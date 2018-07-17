## Neural network base class

import os
import numpy as np

from dnn.nn.utils import _xW

## Base Class
class BaseNN():
	def __init__(self, shape=(4, 2)):
		"""
		param: shape - it is the shape of the input
		 	4 = batch size
			2 = input data length
		Example: shape=(2, 2)
			[[1, 2],
			[3, 4]]
		"""

		self.input_batch_size = shape[0]
		self.input_data_size = shape[1]

		self.has_input = False
		self.last_shape = None

		self.weights = []
		self.activation = []

		self.is_compile = False


	def add(self, num, activation='sigmoid', weight_init='uniform'):
		"""
		## This function add the layer to the neural network

		param: num - The number of layer you want to add to
			the neural network
		param: activation - The activation function to be used
		"""

		if self.has_input == False:
			shape = (self.input_data_size, num)
			W = 2 * np.random.random(shape) -1
			self.has_input = True

		else:
			shape = (self.last_shape, num)
			W = 2 * np.random.random(shape) -1

		self.weights.append(W)
		self.last_shape = num
		self.activation.append(activation)

	def summary(self, side='f'):
		if side == 'f':
			dummy_data = np.random.randn(self.input_batch_size, self.input_data_size)
			numW = len(self.weights)
			x = dummy_data

			print("{:6s} {:10s} {:10s} {:10s} {:10s}".
			format("Layer", "Input", "Weight", "Output", "Activation"))

			for i in range(numW):
				output = _xW(x, self.weights[i])
				print("{:6s} {:10s} {:10s} {:10s} {:10s}".format("  "+str(i+1), str(x.shape),
				str(self.weights[i].shape), str(output.shape), self.activation[i]))
				x = output

	def compile(self, loss='absolute_error', batch_size=32, optimizer='sgd', epochs=1000,
	display_step=100, lr=0.01, validation_split=0.0, metrics=None):
		try:
			self.loss = loss
			self.batch_size = batch_size
			self.optimizer = optimizer

			if epochs < 0:
				print("Training epochs should be greater than zero")
				exit()

			self.epochs = epochs

			if display_step < 0:
				print("Display step should be greater than zero")
				exit()

			self.display_step = display_step
			self.lr = lr
			self.validation_split = validation_split
			self.metrics = metrics

			self.is_compile = True
		except Exception as e:
			print("Exception: ", e)
			exit()

	def check_compile(self):
		## Checking if network is compiled or not
		if self.is_compile == False:
			print("Please run the compile function on the network")
			exit()

	def save_weights(self, path=None):
		try:
			file = open(path, 'bw')
			np.save(file, self.weights)
		except Exception as e:
			print("Exception: ", e)
			exit()

	def load_weights(self, path=None):
		try:
			data = np.load(path)
			self.weights = data
		except Exception as e:
			print("Exception: ", e)
			exit()

	def save_model(self, path=None):
		try:
			data = {}
			data['activation'] = self.activation
			data['weights'] = self.weights
			data['input_batch_size'] = self.input_batch_size
			data['input_data_size'] = self.input_data_size
			data['has_input'] = self.has_input
			data['is_compile'] = self.is_compile
			data['loss'] = self.loss
			data['batch_size'] = self.batch_size
			data['optimizer'] = self.optimizer
			data['epochs'] = self.epochs
			data['display_step'] = self.display_step
			data['lr'] = self.lr
			data['validation_split'] = self.validation_split
			data['metrics'] = self.metrics

			file = open(path, 'bw')
			np.save(file, data)
		except Exception as e:
			print("Exception: ", e)
			exit()

	def load_model(self, path=None):
		try:
			data = np.load(path)
			data = data.tolist()
			self.activation = data['activation']
			self.weights = data['weights']
			self.input_batch_size = data['input_batch_size']
			self.input_data_size = data['input_data_size']
			self.has_input = data['has_input']
			self.is_compile = data['is_compile']
			self.loss = data['loss']
			self.batch_size = data['batch_size']
			self.optimizer = data['optimizer']
			self.epochs = data['epochs']
			self.display_step = data['display_step']
			self.lr = data['lr']
			self.validation_split = data['validation_split']
			self.metrics = data['metrics']
		except Exception as e:
			print("Exception: ", e)
			exit()


if __name__ == "__main__":
	bnn = BaseNN()
	bnn.add(3)
	bnn.add(1)
	bnn.summary()
