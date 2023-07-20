import json
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import matplotlib.pyplot as plt 
from pymatgen.core import Composition
from matminer.featurizers.conversions import StrToComposition
import os
from matminer.featurizers.composition import ElementProperty

file_path = '../..data/SparseXCluster.json'
ncluster=50
featurefile = '../..data/bandgap_features.csv'


if not os.path.exists(featurefile):
    df1 = StrToComposition().featurize_dataframe(df, "formula")
    df1.head()
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df2 = ep_feat.featurize_dataframe(df1, col_id="composition")
    df2.head()
    excluded = ["formula", "composition", "formula", "gap expt"]
    df2.to_csv(featurefile)
else:  #load the features
    df2 = pd.read_csv(featurefile)

df = df2[["composition",property_column]]
df['composition']=df['composition'].map(Composition)

with open(file_path, 'r') as file:
    cluster_ids = json.load(file)


baseline_scores= []


data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gap expt"], 
    target_names=["gap_expt_eV"]
)
data.featurize()

for i, cluster_id in enumerate(range(0,ncluster)):
    target_id_list = cluster_ids[str(cluster_id)]
    # print('cluster_id', cluster_id, target_id_list)
    Xt = df.iloc[target_id_list]
    yt = df["gap expt"].iloc[target_id_list].values

    Xs = df.drop(target_id_list)
    ys = df["gap expt"].drop(target_id_list).values

    
    train_id_list = Xs.index.tolist()
    train,test = data.split([train_id_list,target_id_list])
    
    train.feature_selection(n=-1)

    model = MODNetModel([[['gap_expt_eV']]],
                    weights={'gap_expt_eV':1},
                    num_neurons = [[256], [128], [16], [16]],
                    n_feat = 150,
                    act =  "elu"
                   )

    model.fit(train,
          val_fraction = 0.1,
          lr = 0.0002,
          batch_size = 64,
          loss = 'mae',
          epochs = 100,
          verbose = 1,
         )
    pred = model.predict(test)
    score = np.absolute(pred.values-yt).mean()

    print("src_only score: ", score)
    baseline_scores.append(score)

print(f"modnet avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print("modnet all scores:",baseline_scores)


