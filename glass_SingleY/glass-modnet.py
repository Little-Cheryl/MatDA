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




# y = df2[property_column].values
# excluded = ["formula", "composition", "formula", property_column]
# X = df2.drop(excluded, axis=1)
df = df2[["composition",property_column]]
df['composition']=df['composition'].map(Composition)


# In[6]:


# print(df.head())
# print(df.describe())
# print(X.shape)
# print(X.iloc[0])
# print(y)




# Read the JSON file
with open(file_path, 'r') as file:
    cluster_ids = json.load(file)

# Access the data from the JSON file
# target_id_list = cluster_ids["2"]
# print(target_id_list)


# In[ ]:


# from adapt.instance_based import LDM

num_improvement = 0
num_worse = 0
average_improvement = 0
average_worse = 0
baseline_scores=[]
adapt_scores=[]

# from sklearn.model_selection import train_test_split
# split = train_test_split(range(100), test_size=0.1, random_state=1234)

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

    # test_df = df.iloc[target_id_list].reset_index(drop=True)
    # yt = df["gap expt"].iloc[target_id_list].values

    # train_df = df.drop(target_id_list).reset_index(drop=True)
    # ys = df["gap expt"].drop(target_id_list).values

    # This instantiates the MODData
    # test = MODData(
    #     materials=test_df["composition"], # you can provide composition objects to MODData
    #     targets=test_df["gap expt"], 
    #     target_names=["gap_expt_eV"]
    # )
    # test.featurize()
    # # exit()
    # # print(data)

    # train = MODData(
    #     materials=train_df["composition"], # you can provide composition objects to MODData
    #     targets=train_df["gap expt"], 
    #     target_names=["gap_expt_eV"]
    # )
    # train.featurize()
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
          # "num_classes": {'gfa':2},
          loss = 'binary_crossentropy',
          epochs = 100,
          verbose = 1,
         )
    pred = model.predict(test)
    y_pred = convert_to_boolean(pred.values)
    score = balanced_accuracy_score(yt,y_pred)

    # src_only = RandomForestRegressor(n_estimators=100, max_depth=10)
    # print(f"{i}\t",end="")
    # src_only.fit(Xs, ys)

    # score = src_only.score(Xt, yt)
    print("src_only score: ", score)
    baseline_scores.append(score)
    print("all scores", baseline_scores)
    # exit()
    # Instantiate a KMM model : estimator and target input
    # data Xt are given as parameters with the kernel parameters


    # adapt_model = KMM(
    #     estimator=RandomForestRegressor(n_estimators=100, max_depth=10),
    #     Xt=Xt,
    #     kernel="rbf",  # Gaussian kernel
    #     gamma=1.,     # Bandwidth of the kernel
    #     verbose=0,
    #     random_state=0
    # )
    

    # # Fit the model.
    # adapt_model.fit(Xs, ys)

    # # Get the score on target data
    # adapt_score = adapt_model.score(Xt, yt)
    # print("\tadapt_model score: ", adapt_score)
    # adapt_scores.append(adapt_score)
    
    # if adapt_score < score:
    #     num_improvement += 1
    #     a = -(score-adapt_score)/score
    #     average_improvement += a
    # else: 
    #     num_worse += 1
    #     w = (adapt_score-score)/score
    #     average_worse += w


# print(f"average improvement: {(average_improvement/ncluster)*100} %")
# print(f"average worse: {(average_worse/ncluster)*100} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
# print(f"adapt avg. accuracy:{sum(adapt_scores)/len(adapt_scores)}")
# print(f"num_improvement:{num_improvement}")
# print(f"num_worse:{num_worse}")




# df3 = load_dataset("matbench_expt_gap")
# df3["composition"] = df3["composition"].map(Composition) # maps composition to a pymatgen composition object
# print(df3.head())
# print(df3.describe())



