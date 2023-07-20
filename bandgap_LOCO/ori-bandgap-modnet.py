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
# print(data)
# exit()

from sklearn.model_selection import train_test_split
split = train_test_split(range(100), test_size=0.1, random_state=1234)
train, test = data.split(split)

    
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

mae_test = np.absolute(pred.values-test.df_targets.values).mean()
print(f'mae: {mae_test}')


