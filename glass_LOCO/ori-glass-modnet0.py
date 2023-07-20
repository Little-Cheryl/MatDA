import json
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import matplotlib.pyplot as plt 
from pymatgen.core import Composition

from collections import defaultdict
import itertools
import os
from modnet.featurizers import MODFeaturizer
from modnet.featurizers.presets import DeBreuck2020Featurizer
from sklearn.metrics import balanced_accuracy_score
# In[3]:
import modnet
from modnet.matbench.benchmark import matbench_benchmark

# modnet.__version__
# df = load_dataset("matbench_glass")
# df["composition"] = df["composition"].map(Composition)

# df['gfa'] = df['gfa'].map(int)
# df.describe()

class CompositionOnlyFeaturizer(MODFeaturizer):
    composition_featurizers = DeBreuck2020Featurizer.composition_featurizers
    
    def featurize_composition(self, df):
        """ Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        from pymatgen.core.periodic_table import Element 
        import numpy as np
        from modnet.featurizers import clean_df
        df = super().featurize_composition(df)
        _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
        df['AtomicOrbitals|HOMO_character'] = df['AtomicOrbitals|HOMO_character'].map(_orbitals)
        df['AtomicOrbitals|LUMO_character'] = df['AtomicOrbitals|LUMO_character'].map(_orbitals)

        df['AtomicOrbitals|HOMO_element'] = df['AtomicOrbitals|HOMO_element'].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )
        df['AtomicOrbitals|LUMO_element'] = df['AtomicOrbitals|LUMO_element'].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )

        df = df.replace([np.inf, -np.inf, np.nan], 0)
        
        return clean_df(df)

class CompositionContainer:
    def __init__(self, composition):
        self.composition = composition

PRECOMPUTED_MODDATA = "./precomputed/glass_benchmark_moddata.pkl.gz"

if os.path.isfile(PRECOMPUTED_MODDATA):
    data = MODData.load(PRECOMPUTED_MODDATA)
else:
    # Use a fresh copy of the dataset
    df = pd.read_csv("modnet_glass_dataset.csv")
    df["composition"] = df["composition"].map(Composition)
    df["structure"] = df["composition"].map(CompositionContainer)
    
    data = MODData(
        structures=df["composition"].tolist(), 
        targets=df["gfa"].tolist(), 
        target_names=["gfa"],
        featurizer=CompositionOnlyFeaturizer(n_jobs=8),
        num_classes = {'gfa':2}
    )
    data.featurize()
    # As this is a small data/feature set, order all features 
    # data.feature_selection(n=-1)
    # data.save(PRECOMPUTED_MODDATA)

# try:
#     plot_benchmark
# except:
#     import sys
#     sys.path.append('..')
#     from modnet_matbench.utils import *

from sklearn.model_selection import KFold
#from modnet.models import MODNetModel
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

data.df_targets['gfa'] = data.df_targets['gfa'].map(int)

best_settings = {
    "increase_bs":True,
    "num_neurons": [[128], [64], [16], []],
    "n_feat": 150,
    "lr": 0.002,
    "epochs": 200,
    "verbose": 0,
    "act": "elu",
    "batch_size": 64,
    "num_classes": {'gfa':2},
    "loss": "categorical_crossentropy",
    #"xscale": "standard",
}

results = matbench_benchmark(data, [[["gfa"]]], {"gfa": 1}, best_settings,classification=True, save_folds=True)
print("score:", np.mean(results['scores']))

