## Metrics

import numpy as np

def accuracy(y_true, y_pred):
    return 1.0 - np.mean(np.abs((y_pred - y_true)))

class Metrics():
    def __init__(self):
        pass

    def forward(self, fname, y_true, y_pred):
        if fname == 'accuracy':
            return accuracy(y_true, y_pred)
        else:
            print(fname, " metric if not yet implemented")
            exit()
