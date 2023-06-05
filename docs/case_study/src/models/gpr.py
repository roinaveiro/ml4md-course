import numpy as np
import pandas as pd
import copy

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula
        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
class GPr:

    def __init__(self, kernel='Tanimoto'):  
        
        self.k = kernel

    def fit(self, X_train, y_train):

        if self.k == 'Tanimoto':
          self.kernel = Tanimoto()
        elif self.k == 'rbf':
          # use high dimensional kernel
          self.kernel = gpflow.kernels.SquaredExponential(
              lengthscales=[1.0 for _ in range(X_train.shape[-1])])
        else:
          NotImplementedError('Not implemented kernel.')

        y_train = y_train.reshape(-1, 1)
        X_train = X_train.astype(np.float64)

    
        self.m = gpflow.models.GPR(data=(X_train, y_train), 
                                   mean_function=Constant(np.mean(y_train)), 
                              kernel=self.kernel)
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.m.training_loss, self.m.trainable_variables)


    def predict(self, X_test):

        X_test = X_test.astype(np.float64)
        y_mu, y_var = self.m.predict_f(X_test)


        return y_mu.numpy().squeeze(), np.sqrt(y_var.numpy().squeeze())
