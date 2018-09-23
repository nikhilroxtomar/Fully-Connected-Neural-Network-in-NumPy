## SGD

import numpy as np

def sgd(delta, result, lr):
    """
        SGD: Sochastic Gradient Descent Algorithm

        param: delta -
        param: result - the prediction from the neural network
                [(Input), ((Hidden1), (Hidden2), ..., (Output))]
    """

    numD = len(delta)
    weights = []

    for i in range(numD):
        w = -lr * np.dot(result[-i-2].T, delta[i])
        weights.append(w)

    return weights
