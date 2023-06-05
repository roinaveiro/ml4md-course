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



class ModelEvaluator():

    '''
    Evaluates model to predict Ki

    Parameters
    ----------
    descriptors : vector of name of descriptors to be used
    model       : class cointaining model to be used

    '''
    
    def __init__(self, data_dir, name, feature_set, model):

        self.X, self.X_names, self.y, self.smis = load_task(data_dir, name, 
            feature_set, mask_inputs=True)

        self.model = model

   
    def fit_model(self, data, model, std=False):
        pass


    def evaluate_hold_out(self, n_repeats=10):


        rs = ShuffleSplit(n_splits=n_repeats, test_size=0.33,
            random_state=12462355)
        MAEs = np.zeros(n_repeats)

        p_metric = []
        for train_index, test_index in rs.split(self.X):

            X_train, y_train = self.X[train_index], self.y[train_index]    
            X_test, y_test = self.X[test_index], self.y[test_index]

            self.model.fit(X_train,y_train)
            y_pred = self.model.predict(X_test)
            p_metric.append(metrics.inference_evaluate(y_test, y_pred))

        p_metric = pd.DataFrame(p_metric)
        return p_metric.describe().loc[['mean', '50%', 'std']] 

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
    pass
