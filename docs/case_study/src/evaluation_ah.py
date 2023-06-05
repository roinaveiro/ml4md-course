import pickle 
import numpy as np
import pandas as pd

import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

from config import *

from src.get_descriptors import *
from src.dbs_assembler import * 

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer


import warnings
warnings.filterwarnings("ignore")

class MolEvaluator():


    def __init__(self, filenames, names, dsc, smis_list=None):
        
        self.models = []
        for i, fn in enumerate(filenames):
            self.models.append(pickle.load(
                open(fn, 'rb')))

        self.names = names 
        self.dsc = dsc  # List of descriptors for each model

        if smis_list is not None:
            self.smis_list = smis_list
            self.smis_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True)
             for smi in self.smis_list]
        else:
            gen = Fragments()
            self.smis_list = gen.assemble_dbs()
            self.smis_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True)
             for smi in self.smis_list]


    def get_sascore(self, smis):

        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        s = np.array([sascorer.calculateScore(mol) for mol in mols])

        return s

    def predict_individual(self, smi):

        dfs = {}
        for i, dsc in enumerate(set(self.dsc)):

            dfs[dsc] = ( generate_database(smi, flag=dsc, 
                process=False, fq_p = True, path=None) )
            dfs[dsc].columns = dfs[dsc].columns.map(str)

        pred = np.zeros([len(smi), len(self.models)])
        for i, model in enumerate(self.models):
            X = dfs[self.dsc[i]][model.features_].values 
            pred[:, i] = model.predict(X) 
        #pred[:, -1] = self.get_sascore(smis)

        return pred.squeeze()


    def predict(self, smis):

        dfs = {}
        for i, dsc in enumerate(set(self.dsc)):

            dfs[dsc] = ( generate_database(smis, flag=dsc, 
                process=False, fq_p = True, path=None) )
            dfs[dsc].columns = dfs[dsc].columns.map(str)


        pred = np.zeros([len(smis), len(self.models)])
        for i, model in enumerate(self.models):
            X = dfs[self.dsc[i]][model.features_].values 
            pred[:, i] = model.predict(X) 
        #pred[:, -1] = self.get_sascore(smis)

        return pred


    def generate_pred_df(self, results_path=None):

        pred = self.predict(self.smis_list)
        df = pd.DataFrame(pred, columns=self.names, index=self.smis_list)
        final_df = df.sort_values(by=self.names[0], ascending=False)

        if results_path is not None:
            final_df.to_csv(results_path, index=True)

        return final_df

    def generate_pred_df_chunks(self, results_path=None, size=100):

        chunks =  np.array_split( np.array(self.smis_list), size)
        final_df = pd.DataFrame(columns=self.names)

        for i, chunk in enumerate(chunks):

            print("Working in chunk {:6}  out of {:6}:".format(i, len(chunks)))
            pred = self.predict(chunk)

            print(pred.shape)
            print(len(chunk))
            df = pd.DataFrame(pred, columns=self.names, index=chunk)
       
            final_df = pd.concat([final_df, df])
            
        
        final_df = final_df.sort_values(by=self.names[0], ascending=False)

        if results_path is not None:
            final_df.to_csv(results_path, index=True)

        return final_df




if __name__ == "__main__":

    dsc_list = ['morgan2048', 'modred', 'emb']
    fnames   = []
    names    = []
    dsc_all  = []

    for concentration in [0.2, 0.5]:
        for dsc in dsc_list:
            fnames.append(models_path + f"/millad23_d{dsc}_fqpred_c{concentration}.pkl")
            names.append(f"millad23_d{dsc}_fqpred_c{concentration}")
            dsc_all.append(dsc)

    #smis = pd.read_excel("qt/qt23/CL canonic smiles 20230324.xlsx").values.squeeze()
    #smis = pd.read_excel("qt/qt23/AIB02 analogues virtual screening 20230324.xlsx").values.squeeze()
    smis = list(pd.read_csv("qt/qt23/AIB02_analogues_virtual_screening_20230327.csv").iloc[:,0])
    moleval = MolEvaluator(fnames, names, dsc=dsc_all, smis_list = smis)
    moleval.generate_pred_df_chunks("results/qt23/fpredictions_AIB02_analogues_virtual_screening_20230327.csv", 
                                    size=4)
  
   


    '''
    qt_halogen = pd.read_csv(qt_path + "/halogen_library+cyanide+xylitol_20221115.csv", 
        header=None, skiprows=1)
    qt_full    = pd.read_csv(qt_path + "/full_library_no_halogen+cyanide+xylitol_20221014.csv",
         header=None)

    qt_halogen = qt_halogen.values.squeeze()
    qt_full    = qt_full.values.squeeze()

    fnames = [models_path + "/morgan_fpc_p_c0.2.pkl", models_path + "/morgan_fpc_p_c0.5.pkl"]
    names = ["morgan_fpc_p_c0.2", "morgan_fpc_p_c0.5"]
    
    
    moleval = MolEvaluator(fnames, names, dsc='morgan2048', fq_p=True, 
        smis_list = qt_halogen)

    df_pred = moleval.generate_pred_df_chunks(results_path=
        results_path + "/morgan_fpc_p_c0.5_c0.2_qt_halogen.csv",
        size=50)
    
    moleval = MolEvaluator(fnames, names, dsc='morgan2048', fq_p=True, 
        smis_list = qt_full)

    df_pred = moleval.generate_pred_df_chunks(results_path=
        results_path + "/morgan_fpc_p_c0.5_c0.2_qt_full.csv",
        size=50)
    ''' 
    





   
