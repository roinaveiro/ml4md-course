import os
import pandas as pd
import numpy as np
import ml_collections
import dataclasses

from src.utils import chem, datasets
from config import raw_data_path, mol_col
from config import smi_col, name, target_column, data_dir


def main():

    print(f'Processing data')
    df = pd.read_excel(raw_data_path)

    df = df[df[target_column].notna()]

    df['isosmiles'] = df[smi_col].apply(lambda smi: 
        chem.get_isosmiles(chem.smi_to_mol(smi)))

    df[mol_col] = df['isosmiles'].apply(chem.smi_to_mol)
    not_valid = df[mol_col].isnull()
    print(f'Found {not_valid.sum()} not valid molecules')
    df = df[np.logical_not(not_valid)]
    not_multi = df[mol_col].apply(chem.is_single)
    print(f'Found {np.logical_not(not_multi).sum()} multi-component molecules')
    df = df[not_multi]
    many_atoms = df[mol_col].apply(chem.is_larger_molecule)
    print(f'Found {np.logical_not(many_atoms).sum()} single-atom molecules')
    df = df[many_atoms]

    natoms = df[mol_col].apply(lambda m: m.GetNumAtoms()).values
    natoms_range = (int(np.min(natoms)), int(np.max(natoms)))
    n_mols = len(df['isosmiles'].unique())
    has_unique_smiles = n_mols == len(df)
    print(f'Has unique moles = {has_unique_smiles}')
    atom_set = chem.get_atom_set(df[mol_col])
    print(f'Found the atom_set = {atom_set}')

    relevant_columns = []
    df = df.reset_index()
    relevant_columns.append('index')
    relevant_columns.append(['isosmiles'])
    relevant_columns.append(target_column)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fname = os.path.join(data_dir, f'{name}.csv')
    df.to_csv(fname, index=False)

    new_info = datasets.DatasetInfo(name=name,
                                        atom_set=atom_set,
                                        unique_smiles=has_unique_smiles,
                                        natom_range=natoms_range,
                                        has_salts=False,
                                        n_datapoints=len(df),
                                        n_mols=n_mols,
                                        target_column=target_column,
                                        mol_column=mol_col)

    new_info = ml_collections.ConfigDict(dataclasses.asdict(new_info))
    fname = os.path.join(data_dir, f'{name}.json')
    with open(fname, 'w') as afile:
        afile.write(new_info.to_json())


    

if __name__ == '__main__':
    main()