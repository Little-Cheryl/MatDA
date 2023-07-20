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
from matminer.featurizers.conversions import StrToComposition
import os
import json



featurefile = 'bandgap_features.csv'
property_column='gap expt'




df2 = pd.read_csv(featurefile)
y = df2[property_column].values
excluded = ["formula", "composition", "formula", property_column]
X = df2.drop(excluded, axis=1)



df = df2[["composition",property_column]]
df['composition']=df['composition'].map(Composition)




baseline_scores= []
adapt_scores=[]

# from sklearn.model_selection import train_test_split
# split = train_test_split(range(100), test_size=0.1, random_state=1234)

data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gap expt"], 
    target_names=["gap_expt_eV"]
)
data.featurize()


baseline_scores=[]
baseline_results=[]

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kfold.split(X):
    print("Train index:", train_index)
    print("Test index:", test_index)
    train,test = data.split([train_index,test_index])

    
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
    score = np.absolute(pred.values-test.df_targets.values).mean()


    print("src_only score: ", score)
    baseline_scores.append(score)
    print("all scores", baseline_scores)

print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print("base std:", np.std(baseline_scores))




