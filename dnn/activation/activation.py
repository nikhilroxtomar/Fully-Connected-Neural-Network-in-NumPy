## Activation

from dnn.activation.sigmoid import Sigmoid
from dnn.activation.tanh import Tanh

class Activation():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, fname, x):
        if fname == 'sigmoid':
            return self.sigmoid.forward(x)
        elif fname == 'tanh':
            return self.tanh.forward(x)
        else:
            return '{0} activation function is not supported.'.format(fname)
            exit()

    def backward(self, fname, x):
        if fname == 'sigmoid':
            return self.sigmoid.backward(x)
        elif fname == 'tanh':
            return self.tanh.backward(x)
        else:
            return '{0} activation function is not supported.'.format(fname)
            exit()
