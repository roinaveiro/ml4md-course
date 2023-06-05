import numpy as np
import pandas as pd

from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split

class NGB():

    def __init__(self, n_estimators=2000, lr=0.005, tol=1e-4):

        self.model = NGBRegressor(n_estimators=n_estimators, 
                                  random_state=42, 
                                  learning_rate=lr, tol=tol)

    def fit(self, X_train, y_train, X_val=None, y_val=None, patience=100, verbose=True):
        # self.model.fit(X_train, y_train, X_val, y_val,
        #               early_stopping_rounds=patience)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
            test_size=0.1, random_state=42)

        self.model.fit(X_train, y_train, X_val, y_val,
                     early_stopping_rounds=patience)

    def predict(self, X_test):
        y_dists = self.model.pred_dist(X_test)
        return y_dists.loc, np.sqrt(y_dists.var)

