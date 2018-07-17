## Calculating the loss

from dnn.loss.regression import *

class Loss():
    def __init__(self):
        """
            Differnet types of loss class in a single class
        """
        self.regression_loss = RegressionLoss()

    def forward(self, fname, y_true, y_pred):
        if fname == 'mse':
            return self.regression_loss.mse(y_true, y_pred)
        elif fname == 'absolute_error':
            return self.regression_loss.absolute_error(y_true, y_pred)
        else:
            print(fname, ' loss function is not implented yet.')
            exit()
