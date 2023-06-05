import numpy as np
import pandas as pd

from config import *
from src.utils.datasets import load_task
import src.utils.metrics as metrics

from scipy.stats import sem

from sklearn.model_selection import ShuffleSplit

import rdkit.Chem.AllChem as Chem

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
from src.compute_features import MolFeatures

from sklearn.metrics import r2_score

class MolEvaluator():

    def __init__(self, data_dir, name, features, model, scaler_x=None, scaler_y=None):

        self.features = features
        if features == 'morgan':
            feature_set = ['morgan']
        elif features == 'mordred':
            feature_set = ['mordred']
        else:
            NotImplementedError('Not implemented features')

        self.X, self.X_names, self.y, self.smis = load_task(data_dir, name, 
            feature_set, mask_inputs=True)

        self.y = self.y.reshape(-1, 1)
    
        scaler_y = StandardScaler()
        scaler_y.fit(self.y)
        self.y = scaler_y.transform(self.y)

        self.model = model
        self.model.fit(self.X, self.y)


    def predict_individual(self, smis_list, beta=0.25):

        smis_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True)
            for smi in smis_list]
        mol_feat = MolFeatures(smi_list=smis_list)

        if self.features == 'morgan':
            values, names = mol_feat.get_morgan_fp()
        elif self.features == 'mordred':
            values, names = mol_feat.get_mordred()
        else:
            NotImplementedError('Not implemented features')

        
        tmp_df = pd.DataFrame(values, columns=names)

        # Use UCB 
        mean, sd = self.model.predict(tmp_df[self.X_names].values)
        ucb =  mean + beta*sd
        return ucb


if __name__ == "__main__":

    model = NGB()
    features = 'mordred'
    evaluator = MolEvaluator(data_dir, name, features, model)

    ucb = evaluator.predict_individual(['COC(=O)[C@@H](O)[C@H]1OC(c2ccccc2)O[C@H]2COC(c3ccccc3)O[C@H]21'])

    print(ucb)


