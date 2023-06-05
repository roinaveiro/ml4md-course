import dataclasses
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

import ml_collections
import json
import os

from src.utils import files, chem
from src.utils.files import get_filename, _check_ext, load_npz


@dataclasses.dataclass
class DatasetInfo:
    name: str
    atom_set: str
    unique_smiles: bool
    natom_range: Tuple[int, int]
    has_salts: bool
    n_datapoints: int
    n_mols: int
    target_column: str
    mol_column: str 
    smiles_column: str = 'isosmiles'
    

def load_config(fname: str):
    _check_ext(fname, '.json')
    with open(fname, 'r') as afile:
        config = ml_collections.ConfigDict(json.load(afile))
    return config

def load_dataset(name, work_dir):
    config = load_config(get_filename(name, work_dir, '.json'))
    config = DatasetInfo(**config)
    df = pd.read_csv(os.path.join(work_dir, name + '.csv'))
    df = df.drop_duplicates(config.smiles_column).reset_index(drop=True)
    df['mol'] = df[config.smiles_column].apply(chem.smi_to_mol)
    smiles = np.array(df[config.smiles_column].tolist())
    return df, config, smiles

    
def load_task(data_dir, name, feature_set, mask_inputs=True):

    data = load_npz(f'{data_dir}/{name}_{feature_set[0]}.npz')

    smis = data["isosmiles"]
    y = data["target"]
    X_dsc  = data["dsc_values"]
    # X_prop = data["properties_values"]
    X = X_dsc

    X_names = data["dsc_names"]

    for dsc in feature_set[1:]:
        data = load_npz(f'{data_dir}/{name}_{dsc}.npz')
        X_dsc  = data["dsc_values"]
        X = np.concatenate([X, X_dsc], axis=1)
        X_names = np.concatenate([X_names, data["dsc_names"]])

    if mask_inputs:
        mask = np.logical_and(np.sum(np.isnan(X), axis=0) == 0, np.std(X, axis=0) > 0.0)
        X = X[:, mask]
        print(f'Masking {np.sum(np.logical_not(mask))} feature dims for {feature_set}')

    return X, X_names, y, smis



'''
def load_task(name, feature_set, model,
              data_dir, mask_inputs=True, seed=123, verbose=True):
    
    df, config, smiles = load_dataset(name, data_dir)
    smi_dict = files.load_npz(files.get_filename('random_tvt', data_dir, '.npz'))
    
    y = process_y(df[config.target_columns[:]])

    indices = {}
    for key, smi in smi_dict.items():
        indices[key] = np.array(df[np.isin(smiles, smi)].index.tolist())
        

    split_type = 'random' 


    split = types.IndexSplit(**indices)
    smi_split = types.ArraySplit(smiles, split)

    # decide the scaling based on the feature, model and task
    feature_scaler, target_scaler = scaling_options(feature_set, model)

    if feature_set == 'graph_tuples_graphnet':
        loaded_smi, g = files.load_graphstuple(files.get_filename(feature_set, data_dir, '.npz'))
        smi_to_index = {s: index for index, s in enumerate(loaded_smi)}
        new_indices = np.array([smi_to_index[s] for s in smiles])
        g = graphs.get_graphs(g, new_indices)
        x = graphs.GraphSplit(g, split)
    else:
        feature = utils.SmilesMap(files.get_filename(feature_set, data_dir, '.npz'))
        values = feature(smiles)
        # For removing NAs and constant features
        if mask_inputs:
            mask = np.logical_and(np.sum(np.isnan(values), axis=0) == 0, np.std(values, axis=0) > 0.0)
            values = values[:, mask]
            if verbose:
                print(f'Masking {np.sum(np.logical_not(mask))} feature dims for {feature_set}')

        x = types.ScaledArraySplit(values, split, feature_scaler)

    y = types.ScaledArraySplit(y, split, target_scaler)

    return smi_split, x, y

'''
