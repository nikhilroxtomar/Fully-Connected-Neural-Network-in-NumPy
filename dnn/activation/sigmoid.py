## Sigmoid activation function

import numpy as np

class Sigmoid:
    def __init__(self):
        """
        ## This function is used to calculate the sigmoid and the derivative
            of a sigmoid
        """

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        return x * (1.0 - x)
