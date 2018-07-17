## All the optimizer

import numpy as np
from dnn.optimizer.sgd import sgd

class Optimizer():
    def __init__(self):
        pass

    def optimize(self, fname, delta, result, lr):
        if fname == 'sgd':
            return sgd(delta, result, lr)
        else:
            print(fname, ' optimizer is not implemented yet.')
            exit()
