## Tanh activation function

import numpy as np

class Tanh:
    def __init__(self):
        """
        ## This function is used to calculate the sigmoid and the derivative
            of a tanh
        """

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1.0 - np.square(x)
