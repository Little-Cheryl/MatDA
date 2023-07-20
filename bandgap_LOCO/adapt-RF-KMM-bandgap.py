#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier

# Import KMM method form adapt.instance_based module
from adapt.instance_based import KMM
import json
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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


# In[ ]:





# In[5]:


y = df2[property_column].values
excluded = ["formula", "composition", "formula", property_column]
X = df2.drop(excluded, axis=1)


# In[6]:


# print(X.head)
# print(X.shape)
# print(X.iloc[0])
# print(y)


# In[ ]:





# In[7]:


# from pymatgen.core import Composition

# target_id_list = []
# for i, s in enumerate(df2["formula"]):
#     if "Cu" in Composition(s).as_dict().keys() and "Li" in Composition(s).as_dict().keys():
#         target_id_list.append(i)
# print(target_id_list)


# In[8]:


# Read the JSON file
with open(file_path, 'r') as file:
    cluster_ids = json.load(file)

# Access the data from the JSON file
# target_id_list = cluster_ids["2"]
# print(target_id_list)


# In[ ]:


from adapt.instance_based import LDM

num_improvement = 0
num_worse = 0
average_improvement = 0
average_worse = 0
baseline_scores=[]
adapt_scores=[]

for i, cluster_id in enumerate(range(0,ncluster)):
    target_id_list = cluster_ids[str(cluster_id)]

    Xt = X.iloc[target_id_list]
    yt = df2["gap expt"].iloc[target_id_list].values

    Xs = X.drop(target_id_list)
    ys = df2["gap expt"].drop(target_id_list).values
    
    src_only = RandomForestRegressor(n_estimators=100, max_depth=10)
    print(f"{i}\t",end="")
    src_only.fit(Xs, ys)

    y_pred = src_only.predict(Xt)
    mae = mean_absolute_error(yt, y_pred)
    # r2 = r2_score(yt, y_pred)
    # rmse = np.sqrt(mean_squared_error(yt, y_pred))
    print("src_only mae: ", mae, end="")
    baseline_scores.append(mae)
    score=mae
    
    # Instantiate a KMM model : estimator and target input
    # data Xt are given as parameters with the kernel parameters
    adapt_model = KMM(
        estimator=RandomForestRegressor(n_estimators=100, max_depth=10),
        Xt=Xt,
        kernel="rbf",  # Gaussian kernel
        gamma=1.,     # Bandwidth of the kernel
        verbose=0,
        random_state=0
    )
    

    # Fit the model.
    adapt_model.fit(Xs, ys)

    # Get the score on target data
    y_pred = adapt_model.predict(Xt)

    nan_count = np.sum(np.isnan(y_pred))
    if nan_count !=0:
        #print("Number of NaN values:", nan_count)
        #print(y_pred)
        #print(Xt.shape)
        print("\tadapt_model mae: ", 'NaN', end="")
        fails+=1
        continue

    mae = mean_absolute_error(yt, y_pred)
    print("\tadapt_model mae: ", mae, end="")
    adapt_score = mae
    adapt_scores.append(adapt_score)
    
    if adapt_score < score:
        num_improvement += 1
        a = -(score-adapt_score)/score
        average_improvement += a
    elif adapt_score > score: 
        num_worse += 1
        w = (adapt_score-score)/score
        average_worse += w
    # print("_"*50)
    print()

print(f"average improvement: {(average_improvement/ncluster)*100} %")
print(f"average worse: {(average_worse/ncluster)*100} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print(f"adapt avg. accuracy:{sum(adapt_scores)/len(adapt_scores)}")
print(f"num_improvement:{num_improvement}")
print(f"num_worse:{num_worse}")

print("base std:",np.std(baseline_scores))
print("adapt std:",np.std(adapt_scores))
