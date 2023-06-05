import numpy as np
import os
import json

import itertools
from absl import app, flags



def _check_ext(fname: str, ext: str):
    if not fname.endswith(ext): 
        raise ValueError(f'Expected extension "{ext}" in {fname}')

def get_filename(name, work_dir, ext):
    return os.path.join(work_dir, name + ext)

def load_npz(fname):
    _check_ext(fname, 'npz')
    data = np.load(fname, allow_pickle=True)
    return {key: data[key] for key in data.keys()}
