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
import modnet

file_path = '../..data/SparseXSingle.json'
ncluster=50
featurefile = '../..data/glass_features.csv'


if not os.path.exists(featurefile):
    df1 = StrToComposition().featurize_dataframe(df, "formula")
    df1.head()
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df2 = ep_feat.featurize_dataframe(df1, col_id="composition")
    df2.head()
    excluded = ["formula", "composition", "formula", "gfa"]
    df2.to_csv(featurefile)
else:  #load the features
    df2 = pd.read_csv(featurefile)

df = df2[["composition",'gfa']]
df['composition']=df['composition'].map(Composition)

with open(file_path, 'r') as file:
    cluster_ids = json.load(file)

def convert_to_boolean(lst):
    boolean_list = [bool(element) for element in lst]
    return boolean_list


baseline_scores=[]


data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gfa"], 
    target_names=["gfa"],
    num_classes = {'gfa':2}
)
data.featurize()

for i, cluster_id in enumerate(range(0,ncluster)):
    target_id_list = cluster_ids[str(cluster_id)]
    print('cluster_id', cluster_id, target_id_list)

    Xt = df.iloc[target_id_list]
    yt = df["gfa"].iloc[target_id_list].values

    Xs = df.drop(target_id_list)
    ys = df["gfa"].drop(target_id_list).values

    train_id_list = Xs.index.tolist()
    train,test = data.split([train_id_list,target_id_list])
    
    train.feature_selection(n=-1)

    model = MODNetModel([[['gfa']]],
                    weights={'gfa':1},
                    num_neurons = [[128], [64], [16], []],
                    n_feat = 150,
                    act =  "elu",
                   )

    model.fit(train,
          val_fraction = 0.1,
          lr = 0.0002,
          batch_size = 64,
          loss = 'binary_crossentropy',
          epochs = 100,
          verbose = 1,
         )
    pred = model.predict(test)
    y_pred = convert_to_boolean(pred.values)
    score = balanced_accuracy_score(yt,y_pred)

    print("src_only score: ", score)
    baseline_scores.append(score)
    
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print("adapt scores", baseline_scores)


