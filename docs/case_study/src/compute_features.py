import numpy as np
import pandas as pd

import rdkit.Chem.AllChem as Chem
from mordred import Calculator, descriptors

from src.utils.datasets import load_dataset
from src.utils import chem

from config import name, data_dir

class MolFeatures():

    def __init__(self, name=None, data_dir=None, smi_list=None, nunique=False):

        self.nunique = nunique # Drop columns with unique values

        if smi_list is not None:
            self.mol = [chem.smi_to_mol(smi) for smi in smi_list]

        else:
            self.name  = name
            self.df, self.config, self.smiles = load_dataset(name, data_dir)
            print(self.df.head())
            self.mol = self.df[self.config.mol_column].tolist()


    def get_morgan_fp(self):

        # print("Getting Morgan Fingerprints")
        values = [np.array( 
            Chem.GetMorganFingerprintAsBitVect(mol, 3, 2048) 
            ) for mol in self.mol]

        df = pd.DataFrame(values)

        if self.nunique:
            nunique = df.nunique()
            cols_to_drop = nunique[nunique == 1].index
            df = df.drop(cols_to_drop, axis=1)
        
        # Drop columns with NA
        df = df.dropna(axis='columns')

        return df.values, df.columns

    def get_mordred(self):

        # print("Getting Mordred Descriptors")
        calc = Calculator(descriptors, ignore_3D=True)
        df = calc.pandas(self.mol, nproc=1).fill_missing(value=np.nan)

        if self.nunique:
            nunique = df.nunique()
            cols_to_drop = nunique[nunique == 1].index
            df = df.drop(cols_to_drop, axis=1)
        
        # Drop columns with NA
        df = df.dropna(axis='columns')
        df["Lipinski"] = df["Lipinski"].astype(float)
        df["GhoseFilter"] = df["GhoseFilter"].astype(float)

        return df.values, df.columns

    def write_descriptors(self, data_dir):
        
        morgan_values, morgan_names  = self.get_morgan_fp()
        fname = f'{data_dir}/{self.name}_morgan.npz'
        np.savez_compressed(fname, isosmiles=self.smiles, 
                                   target=self.df[self.config.target_column],
                                   dsc_names=morgan_names,
                                   dsc_values=morgan_values)

        mordred_values, mordred_names  = self.get_mordred()
        fname = f'{data_dir}/{self.name}_mordred.npz'
        np.savez_compressed(fname, isosmiles=self.smiles, 
                                   target=self.df[self.config.target_column],
                                   dsc_names=mordred_names,
                                   dsc_values=mordred_values)



if __name__ == '__main__':

    mol_feat = MolFeatures(smi_list=['CCC'])
    values, names = mol_feat.get_morgan_fp()
    values, names = mol_feat.get_mordred()
    print(values)

    # mol_feat.write_descriptors(data_dir)



'''


def get_embedding(smile_input, tokenizer, model):
    """
    input is a single string
    """
    input_sentence = torch.tensor(tokenizer.encode(smile_input)).unsqueeze(0)
    out = model(input_sentence, output_hidden_states=True)
    return out[1][-1][0, :, :].mean(dim=0).detach().numpy()

def get_embedding_df(smis):

    #tokenizer = AutoTokenizer.from_pretrained("mrm8488/chEMBL26_smiles_v2")  
    #model = AutoModelForMaskedLM.from_pretrained("mrm8488/chEMBL26_smiles_v2")

    #from transformers import pipeline, RobertaModel, RobertaTokenizer

    # model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    # tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

    model = AutoModelForMaskedLM.from_pretrained("DeepChem/SmilesTokenizer_PubChem_1M", ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/SmilesTokenizer_PubChem_1M")

    # tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")  
    # model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    embs = np.vstack( pd.Series(smis).apply(lambda x: 
            get_embedding(x, tokenizer, model)) )
    df = pd.DataFrame(embs)

    return df

def get_modred_descriptors(smis):


    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smis]

    df = calc.pandas(mols,nproc=7).fill_missing(value=np.nan)
    df["Lipinski"] = df["Lipinski"].astype(float)
    df["GhoseFilter"] = df["GhoseFilter"].astype(float)

    df = df.fillna(0)

    return df



def get_fq_descriptors(smis):

    print("Using FQ model:", model_fq_path)
    properties = ['den', 'bp', 'fp', 'sol', 'st', 'vp', 'vis', 'mp']

    with open(model_fq_path, 'rb') as file:
        models = pickle.load(file)

    values = np.stack([[models[p].predict_from_smile(smile) 
        for smile in smis] for p in properties])

    df = pd.DataFrame(np.squeeze(values).T, columns=properties, 
        index=smis)

    return df

def get_fq_descriptors_own(smis):

    properties = ["BP", "Density", "FP", "MP", "Solubility", "Viscosity",
                  "ST", "VP"]

    models = []
    for i in range( len(models_fq_own) ):
        pt = models_path +  models_fq_own[i]
        models.append( pickle.load(open(pt, 'rb')) )

    # X = get_morgan_2048_fp(smis)

    X = get_modred_descriptors(smis)

    preds  = np.zeros( [X.shape[0], len(models)] )

    for i in range(len(properties)):
        XX = X[models[i].descriptors]
        preds[:, i] = models[i].predict(XX)

    df = pd.DataFrame(preds, columns=properties, 
        index=smis)

    return df


def generate_database(smis, flag='modered', process=True, fq_p = False, path=None, 
        na_subs=False):

    if flag == 'modred':
        df = get_modred_descriptors(smis)

    elif flag == 'emb':
        df = get_embedding_df(smis)

    elif flag == 'morgan':
        df = get_morgan_fp(smis)

    elif flag == 'morgan2048':
        df = get_morgan_2048_fp(smis)

    elif flag == 'modred+morgan':
        df1 = get_morgan_2048_fp(smis)
        df2 = get_modred_descriptors(smis)
        df = pd.concat([df1, df2], axis=1)

    elif flag == 'modred+morgan+emb':
        df1 = get_morgan_2048_fp(smis)
        df2 = get_modred_descriptors(smis)
        df3 = get_embedding_df(smis)
        df = pd.concat([df1, df2, df3], axis=1)



    # df = pd.concat( [desc, embs], axis=1 )
    
    if process:
        # Remove constant columns
        nunique = df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)

        # Remove duplicates
        df = df.T.drop_duplicates(keep='first').T
        
        # Drop columns with NA
        df = df.dropna(axis='columns')

    else:
        pass

    if fq_p:

        df_fq = get_fq_descriptors_own(smis)
        df.insert(loc=0,  column='BP',         value=df_fq["BP"].values )
        df.insert(loc=1,  column='Density',    value=df_fq["Density"].values )
        df.insert(loc=2,  column='MP',         value=df_fq["MP"].values )
        df.insert(loc=3,  column='Solubility', value=df_fq["Solubility"].values )
        df.insert(loc=4, column='ST',          value=df_fq["ST"].values )
        df.insert(loc=5, column='Viscosity',  value=df_fq["Viscosity"].values )
        df.insert(loc=6, column='VP',         value=df_fq["VP"].values )
        df.insert(loc=7, column='FP',         value=df_fq["FP"].values )

    else:
        pass

    if na_subs:
        df = df.fillna(0)
        


    if path:
        df.to_csv(path, index=False)

    return df
 
    


if __name__ == "__main__":

    dsc_list = ['morgan2048', 'modred', 'emb', 'modred+morgan', 'modred+morgan+emb']

    for concentration in [0.5]:
        for dsc in dsc_list:

            fname = f"millad23_d{dsc}_fqpred_c{concentration}.csv"

            print("Concentration: ", concentration)

            # data = pd.read_csv(data_path + "/data23.csv")
            data = pd.read_excel(data_path + "/data23.xlsx")

            smis = data.can_SMILES.values
            smis_can = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in smis]

            df = generate_database(smis_can, flag=dsc, process=True, fq_p=True)


            if concentration == 0.2:
                df.insert(loc=0, column='label',             value=data["label"].values )
                df.insert(loc=1, column='can_SMILES',        value=smis_can )
                #df.insert(loc=2, column='Tcp',               value=data["Tcp_2"].values )
                #df.insert(loc=3, column='clarity',           value=data["clarity_2"].values )
                df.insert(loc=2, column='anti-haze',         value=data["anti-haze2"].values )

            elif concentration == 0.5:
                df.insert(loc=0, column='label',             value=data["label"].values )
                df.insert(loc=1, column='can_SMILES',        value=smis_can )
                #df.insert(loc=2, column='Tcp',               value=data["Tcp_5"].values )
                #df.insert(loc=3, column='clarity',           value=data["clarity_5"].values )
                df.insert(loc=2, column='anti-haze',         value=data["anti-haze5"].values )

                    
            df["anti-haze"][df["anti-haze"] < 0.0] = 0.0
            df.to_csv(preprocessed_path + "/" + fname, index=False)

if fq == 'chinese':
                df = generate_database(smis_can, flag=dsc)
                df.insert(loc=0,  column='BP',         value=data["BP"].values )
                df.insert(loc=1,  column='Density',    value=data["Density"].values )
                df.insert(loc=2,  column='MP',         value=data["MP"].values )
                df.insert(loc=3,  column='Solubility', value=data["Solubility"].values )
                df.insert(loc=4, column='ST',         value=data["ST"].values )
                df.insert(loc=5, column='Viscosity',  value=data["Viscosity"].values )
                df.insert(loc=6, column='VP',         value=data["VP"].values )
                df.insert(loc=7, column='FP',         value=data["FP"].values )

            elif fq == 'epibat':
                df = generate_database(smis_can, flag=dsc)
                df.insert(loc=0,  column='BP',         value=data["BP_e"].values )
                df.insert(loc=1,  column='Density',    value=data["Density"].values )
                df.insert(loc=2,  column='MP',         value=data["MP_e"].values )
                df.insert(loc=3,  column='Solubility', value=data["Solubility"].values )
                df.insert(loc=4, column='ST',         value=data["ST"].values )
                df.insert(loc=5, column='Viscosity',  value=data["Viscosity"].values )
                df.insert(loc=6, column='VP',         value=data["VP_e"].values )
                df.insert(loc=7, column='FP',         value=data["FP"].values )

'''