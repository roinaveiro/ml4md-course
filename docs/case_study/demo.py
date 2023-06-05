
from config import data_dir, name
from src.evaluator import MolEvaluator

from src.models.ngb import NGB
from src.models.gpr import GPr

from src.evomol import run_model
run_model({
    "obj_function": "anti-haze",
    "optimization_parameters": {
        "max_steps": 500
    },
    "optimization_parameters": {
        "mutable_init_pop": False
    },
    "io_parameters": {
        "smiles_list_init" : ['CCCc1ccc(C2O[C@@H]3[C@@H](OC(c4ccc(CCC)cc4)O[C@@H]3[C@H](O)CO)[C@@H](CCC)O2)cc1'],
        "model_path": "partial_results/test_sascore"
    },
})