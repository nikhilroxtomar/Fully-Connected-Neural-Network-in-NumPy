## Regression Loss

class RegressionLoss:
    def __init__(self):
        pass

    def mse(self, y_true, y_pred):
        return (y_pred - y_true)**2

    def absolute_error(self, y_true, y_pred):
         return y_pred - y_true
