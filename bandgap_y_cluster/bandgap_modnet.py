#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from modnet.models import MODNetModel
from modnet.preprocessing import MODData
import matplotlib.pyplot as plt 
from pymatgen.core import Composition
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neural_network import MLPClassifier

# Import KMM method form adapt.instance_based module
# from adapt.instance_based import KMM
import json


# with open('matbench_expt_is_metal.json', 'r') as json_file:
#     # Load the JSON data
#     data = json.load(json_file)

# df = pd.DataFrame(data['data'], columns = data['columns'])
# df = df.rename(columns={'composition': 'formula'})


# In[3]:


# Define the file path
file_path = 'bandgap_target_clusters50.json'
ncluster=50
featurefile = 'bandgap_features.csv'
property_column='gap expt'


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
baseline_scores= [1.997164021945466, 1.391371614798997, 0.42868607225097904, 0.15764798477390518, 1.394652263361754, 1.130300466433944, 1.1665006752294305, 0.04803238627190974, 0.09385822352825522, 0.38021874945423834, 1.5428834096729753, 0.7518992374453584, 1.5975557865196088, 1.3567483358708317, 0.2642086673998274, 1.1714173531457859, 0.06661054076090489, 0.3007527493660401, 2.190518129888841, 2.129888086230016, 1.3994530388100337, 0.04711815031866232, 0.1286597384837886, 0.20912066612156985, 0.7048445761569646, 1.4985486875906182, 1.3245240241752003, 0.27809325245589744, 1.136391640141599, 0.3243329683570629, 0.5599891035835995, 1.8117871999831834, 1.3230813260577707, 0.45503228985601, 2.580714202461063, 0.328500778901577, 0.08722474510355425, 0.06657027039622729, 0.14621382803390068, 0.006747401203028858, 0.8558189520941807, 1.6620748041325657, 0.19088317874781024, 1.7735676625668002, 1.646986777751161, 1.394724376655856, 0.3693856797898988]
adapt_scores=[]

# from sklearn.model_selection import train_test_split
# split = train_test_split(range(100), test_size=0.1, random_state=1234)

data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gap expt"], 
    target_names=["gap_expt_eV"]
)
data.featurize()
# print(data)
# exit()

for i, cluster_id in enumerate(range(47,ncluster)):
    target_id_list = cluster_ids[str(cluster_id)]
    print('cluster_id', cluster_id, target_id_list)
    Xt = df.iloc[target_id_list]
    yt = df["gap expt"].iloc[target_id_list].values

    Xs = df.drop(target_id_list)
    ys = df["gap expt"].drop(target_id_list).values

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

    # src_only = RandomForestRegressor(n_estimators=100, max_depth=10)
    # print(f"{i}\t",end="")
    # src_only.fit(Xs, ys)

    # score = src_only.score(Xt, yt)
    print("src_only score: ", score)
    baseline_scores.append(score)
    print(" all score: ",baseline_scores)
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



