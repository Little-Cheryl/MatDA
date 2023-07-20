#!/usr/bin/env python
# coding: utf-8

# In[6]:


import json
import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier

# Import KMM method form adapt.instance_based module
# from adapt.instance_based import KMM
import json

datafile = 'matbench_expt_is_metal.json'
datafile = 'matbench_glass.json'
datafile = 'matbench_expt_gap.json'

feature_file = 'glass_features.csv'
feature_file = 'bandgap_features.csv'

with open(datafile, 'r') as json_file:
    # Load the JSON data
    data = json.load(json_file)

df = pd.DataFrame(data['data'], columns = data['columns'])
df = df.rename(columns={'composition': 'formula'})


# In[ ]:





# In[7]:


from matminer.featurizers.conversions import StrToComposition
import os

if not os.path.exists(feature_file):
    df1 = StrToComposition().featurize_dataframe(df, "formula")
    df1.head()

    from matminer.featurizers.composition import ElementProperty

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df2 = ep_feat.featurize_dataframe(df1, col_id="composition")  # input the "composition" column to the featurizer
    df2.head()

    #y = df2['is_metal'].values
    #excluded = ["formula", "composition", "formula", "is_metal"]
    #y = df2['gfa'].values
    #excluded = ["formula", "composition", "formula", "gfa"]

    #X = df2.drop(excluded, axis=1)
    df2.to_csv(feature_file)
else:  #load the features
    df2 = pd.read_csv(feature_file)

print(df2.shape)
print(df2.head())


# In[8]:


# print(X.head)
# print(X.shape)
# print(X.iloc[0])
# print(y)


# In[9]:


# Read the JSON file
# with open(file_path, 'r') as file:
#     cluster_ids = json.load(file)

# # Access the data from the JSON file
# target_id_list = cluster_ids["2"]
# print(target_id_list)


# In[10]:


# from sklearn.model_selection import train_test_split


# Xs, Xt, ys, yt = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:




