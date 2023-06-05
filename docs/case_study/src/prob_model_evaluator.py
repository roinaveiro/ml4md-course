import numpy as np
import pandas as pd

from config import *
from src.utils.datasets import load_task
import src.utils.metrics as metrics

from scipy.stats import sem

from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings("ignore")

import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pickle

from sklearn.preprocessing import QuantileTransformer, StandardScaler

###
from config import *
from src.models.ngb import NGB
from src.models.gpr import GPr


class ModelEvaluator():

    '''
    Evaluates model to predict Ki

    Parameters
    ----------
    data_dir : 
    name : 
    feature_set :
    model :
    '''
    
    def __init__(self, data_dir, name, feature_set, model,
        scaler_x = None, scaler_y = None):

        self.X, self.X_names, self.y, self.smis = load_task(data_dir, name, 
            feature_set, mask_inputs=True)

        self.model = model

        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
   
    def fit_model(self, data, model, std=False):
        pass


    def evaluate_hold_out(self, n_repeats=10):


        rs = ShuffleSplit(n_splits=n_repeats, test_size=0.33,
            random_state=235146)
        MAEs = np.zeros(n_repeats)

        p_metric = []
        c_metric = []
        qs  = []
        Cqs = []

        for train_index, test_index in rs.split(self.X):

            X_train, y_train = self.X[train_index], self.y[train_index]    
            X_test, y_test = self.X[test_index], self.y[test_index]

            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

            if self.scaler_x == "quantile":
                n_quantiles = int(X_train.shape[0] / 20)
                scaler = QuantileTransformer(n_quantiles=n_quantiles)
                scaler.fit(X_train)

                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # print(X_test_scaled)


            elif self.scaler_x == "standard":
                scaler = StandardScaler()
                scaler.fit(X_train)

                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # print(X_test_scaled)
                

            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            if self.scaler_y == "standard":
                scaler = StandardScaler()
                scaler.fit(y_train)
                
                y_train_scaled = scaler.transform(y_train)
                y_test_scaled = scaler.transform(y_test)
            else:
                y_train_scaled = y_train
                y_test_scaled = y_test


            self.model.fit(X_train_scaled, y_train_scaled.squeeze())
            y_pred, y_err = self.model.predict(X_test_scaled)


            p_metric.append(metrics.inference_evaluate(y_test_scaled.squeeze(), y_pred))

            cm, qs_i, Cqs_i = metrics.calibration_evaluate(y_test_scaled.squeeze(), y_pred, y_err)
            c_metric.append(cm)
            qs.append(qs_i)
            Cqs.append(Cqs_i)

            

        p_metric = pd.DataFrame(p_metric)
        c_metric = pd.DataFrame(c_metric)

        t1 = p_metric.describe().loc[['mean', '50%', 'std']]
        t2 = c_metric.describe().loc[['mean', '50%', 'std']]

        return(pd.concat([t1, t2], axis=1), pd.DataFrame(qs).describe().loc[["mean", "std"] ],
         pd.DataFrame(Cqs).describe().loc[["mean", "std"] ])

    '''
    def search_params(self, models, std=False, save_best=True, filename="/model.pkl", 
        n_splits=5, n_repeats=10):

        maes = np.zeros(len(models))

        for i, model in enumerate(models):
            print("Working on model number:", i)
            maes[i], _  = self.rKfold(model, std, n_splits, n_repeats)
            print(maes[i])

        if save_best:

            i_best = np.argmin(maes)
            pth = models_path + filename
            best_model = self.fit_model(self.data, models[i_best], std=False)

            best_model.dsc_ = self.dsc
            best_model.fq_p = self.fq_p

            # Check
            if std:
                best_model.predict = lambda x: self.predict_ah(x, best_model, std)

            best_model.mae_estimate_ = maes[i_best]
            pickle.dump(best_model, open(pth, 'wb'))

            print("Best MAE: ", maes[i_best])
            print("Best model: ", models[i_best])

    '''
    

if __name__ == "__main__":

    model = GPr(kernel='Tanimoto')
    feature_set = ["morgan"]
    neval = ModelEvaluator(data_dir, name, feature_set, model)

    m, a, b = neval.evaluate_hold_out(n_repeats=10)

    print(m)

    model = NGB()
    feature_set = ["mordred"]
    neval = ModelEvaluator(data_dir, name, feature_set, model, scaler_x=None, scaler_y=None)

    m, a, b = neval.evaluate_hold_out(n_repeats=10)

    print(m)


   