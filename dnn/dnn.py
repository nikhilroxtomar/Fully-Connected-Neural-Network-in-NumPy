## The main file

__author__ = "Nikhil Tomar"
__email__ = "nktomar39@gmail.com"
__description__ = "A simple feedforward neural network based on numpy."

from dnn.nn.network import NN as nn

class DNN(nn):
    def __init__(self, shape=(4, 2)):
        """
            Deep Neural Network
        """

        nn.__init__(self, shape=shape)
