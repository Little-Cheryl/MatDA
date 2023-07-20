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
# modnet.__version__

# Define the file path
file_path = 'glass_target_clusters50.json'
ncluster=50
featurefile = 'glass_features.csv'
property_column='gfa'

def convert_to_boolean(lst):
    boolean_list = [bool(element) for element in lst]
    return boolean_list
# In[4]:


from matminer.featurizers.conversions import StrToComposition
import os

if not os.path.exists(featurefile):
    df1 = StrToComposition().featurize_dataframe(df, "formula")
    df1.head()

    from matminer.featurizers.composition import ElementProperty

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df2 = ep_feat.featurize_dataframe(df1, col_id="composition")  # input the "composition" column to the featurizer
    df2.head()

    #y = df2['gfa'].values
    excluded = ["formula", "composition", "formula", property_column]

    #X = df2.drop(excluded, axis=1)
    df2.to_csv(featurefile)
else:  #load the features
    df2 = pd.read_csv(featurefile)


df = df2[["composition",property_column]]
df['composition']=df['composition'].map(Composition)



# Read the JSON file
with open(file_path, 'r') as file:
    cluster_ids = json.load(file)


baseline_scores=[]


data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gfa"], 
    target_names=["gfa"],
    num_classes = {'gfa':2}
)
data.featurize()
# print(data)
# exit()

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
          lr = 0.002,
          batch_size = 64,
          # "num_classes": {'gfa':2},
          loss = 'binary_crossentropy',
          epochs = 200,
          verbose = 0,
         )
    pred = model.predict(test)
    y_pred = convert_to_boolean(pred.values)
    score = balanced_accuracy_score(yt,y_pred)


    print("src_only score: ", score)
    baseline_scores.append(score)
    print("all scores", baseline_scores)



# print(f"average improvement: {(average_improvement/ncluster)*100} %")
# print(f"average worse: {(average_worse/ncluster)*100} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print("base std:", np.std(baseline_scores))




