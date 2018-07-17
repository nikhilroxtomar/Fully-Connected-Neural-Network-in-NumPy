## Neural Network main class

import numpy as np
import time

from dnn.nn.base_network import BaseNN as bnn
from dnn.nn.utils import _xW, _time
from dnn.loss.loss import Loss
from dnn.activation.activation import Activation
from dnn.optimizer.optimizer import Optimizer
from dnn.metrics.metrics import Metrics

class NN(bnn):
    def __init__(self, shape=(1, 2)):
        bnn.__init__(self, shape=shape)
        self.act_class = Activation()
        self.loss_class = Loss()
        self.optimizer_class = Optimizer()
        self.metrics_class = Metrics()

    def _predict(self, X, side='f'):
        ## Check if network is compiled or not
        self.check_compile()
        result = []
        n = X.shape[0]
        self.input_batch_size = n

        for i in range(len(self.weights)):
            output = _xW(X, self.weights[i])
            output = self.act_class.forward(self.activation[i], output)
            result.append(output)
            X = output
        return result

    def predict(self, X):
        return self._predict(X)[-1]

    def _train(self, X, Y):
        """
            This function is used to train the neural network on fix size dataset

            param: X - input of neural network
            param: Y - output of the neural network
        """
        ## Check if network is compiled or not
        self.check_compile()

        result = self._predict(X)
        result.insert(0, X)
        """
            result = [(Input), ((Hidden1), (Hidden2), ..., (Output))]
        """

        delta = []

        output_error = self.loss_class.forward(self.loss, Y, result[-1])
        output_delta = output_error * self.act_class.backward(self.activation[-1], result[-1])
        delta.append(output_delta)

        numW = len(self.weights)
        for i in range(1, numW):
            #print("Step: ", i, delta[-1].shape, self.weights[-i].T.shape)
            hidden_error = np.dot(delta[-1], self.weights[-i].T)
            hidden_delta = hidden_error * self.act_class.backward(self.activation[-i-1], result[-i-1])
            delta.append(hidden_delta)

        dW = self.optimizer_class.optimize(self.optimizer, delta, result, self.lr)

        for i in range(1, numW+1):
            self.weights[-i] += dW[i-1]

        return np.mean(output_error)

    def train(self, X, Y):
        try:
            if X.shape[0] != Y.shape[0]:
                print("Training data doesn't having matching shape {0}, {1}".format(X.shape, Y.shape))
                exit()

            ## Starting time of training
            start_time = time.time()

            ## Spliting the dataset into the training and validation datset
            validation_size = int(self.validation_split * X.shape[0])
            validation_X = X[0:validation_size]
            validation_Y = Y[0:validation_size]

            X = X[validation_size:]
            Y = Y[validation_size:]

            for step in range(self.epochs):
                one_batch = int(X.shape[0]/self.batch_size)
                self.input_batch_size = one_batch * self.batch_size
                """
                    Here we divide the complete dataset into the mini batches.
                    If total dataset = 120
                    If batch_size = 30
                        then:
                        one_batch = int(120/30) = 4
                        for i in 4:
                            start_idx = i * batch_size
                            end_idx = start_idx + batch_size
                            ## start_idx = 0 * 30 = 0
                            ## end_idx = 0 + 30 = 30
                        [0:30], [30:60], [60:90], [90:20]: perfect batches

                    But if batch_size = 32
                        then:
                        one_batch = int(120/32) = 3
                        for i in 3:
                            start_idx = i * batch_size
                            end_idx = start_idx + batch_size
                            ## start_idx = 0 * 32 = 0
                            ## end_idx = 0 + 32 = 32
                        [0:32], [32:64], [64:96]: perfect batches
                        [96:120]                : inperfect batches

                """
                ## Mini-batch training
                batch_error = []
                for i in range(one_batch):
                    start_idx = i * self.batch_size
                    batch_X = X[start_idx:start_idx+self.batch_size]
                    batch_Y = Y[start_idx:start_idx+self.batch_size]
                    error = self._train(batch_X, batch_Y)
                    batch_error.append(error)

                ## Direct training
                if self.batch_size * one_batch != X.shape[0]:
                    self.input_batch_size = X.shape[0] - one_batch*self.batch_size
                    start_idx = one_batch*self.batch_size
                    batch_X = X[start_idx:X.shape[0]]
                    batch_Y = Y[start_idx:X.shape[0]]
                    error = self._train(batch_X, batch_Y)
                    batch_error.append(error)

                if (step+1) % self.display_step == 0:
                    output_string = ""
                    output_string += "Epoch: {:4d}".format(step+1)
                    batch_error = np.array(batch_error, dtype='float')
                    e = np.mean(batch_error)
                    output_string += "  Error: " + str(e)[:12]

                    ## Validation Dataset
                    if validation_size > 0:
                        validation_output = self.predict(validation_X)
                        val_result = self.metrics_class.forward(self.metrics, validation_Y, validation_output)
                        output_string += "  Val Acc: " + str(val_result)[:8]

                    if self.metrics != None:
                        ## Training Accuracy
                        training_output = self.predict(X)
                        training_result = self.metrics_class.forward(self.metrics, Y, training_output)
                        output_string += "  Train Acc: " + str(training_result)[:8]

                    print(output_string)
            end_time = time.time()
            total_time = _time(end_time - start_time)

            print("Training Finished in ", total_time)

        except Exception as e:
            print("Exception: ", e)
            exit()

    def evaluate(self, X, Y):
        result = []
        output = self.predict(X)
        return self.metrics_class.forward('accuracy', Y, output)
