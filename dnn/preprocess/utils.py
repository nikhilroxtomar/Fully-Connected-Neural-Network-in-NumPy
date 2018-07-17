
import numpy as np

## To Onehot
def category_to_onehot(Y):
    #param: Y - the output of the network

    num = Y.shape[0]
    unique_category = []
    for i in range(num):
        y = int(Y[i])
        if y not in unique_category:
            unique_category.append(y)

    unique_category.sort()
    output = np.zeros((num, len(unique_category)))
    for i in range(num):
        onehot_n = Y[i][0]
        output[i][onehot_n] = 1

    return output
