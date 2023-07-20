#!/usr/bin/env python
# coding: utf-8

# In[2]:
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge

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
# print(f"target_id_list: {len(target_id_list)}")


# In[ ]:


from adapt.instance_based import LDM
from adapt.instance_based import KLIEP
from adapt.instance_based import ULSIF
from adapt.instance_based import RULSIF
from adapt.instance_based import NearestNeighborsWeighting
from adapt.instance_based import WANN
from adapt.parameter_based import LinInt
from adapt.parameter_based import RegularTransferLR

from adapt.parameter_based import RegularTransferNN


num_improvement = 0
num_worse = 0
average_improvement = 0
average_worse = 0
baseline_scores=[]
adapt_scores=[]
fails=0
for i, cluster_id in enumerate(range(0,ncluster)):
    target_id_list = cluster_ids[str(cluster_id)]

    Xt = X.iloc[target_id_list]
    yt = df2["gap expt"].iloc[target_id_list].values

    Xs = X.drop(target_id_list)
    ys = df2["gap expt"].drop(target_id_list).values

    


    
    src_only = RegularTransferNN(loss="mse", lambdas=0., random_state=0)
    print(f"{i}\t",end="")
    src_only.fit(Xs, ys, epochs=100, verbose=0)

    # score = src_only.score(Xt, yt)
    #print("src_only score: ", score, end="")
    #baseline_scores.append(score)

    y_pred = src_only.predict(Xt)
    mae = mean_absolute_error(yt, y_pred)
    # r2 = r2_score(yt, y_pred)
    # rmse = np.sqrt(mean_squared_error(yt, y_pred))
    print("src_only mae: ", mae, end="")
    baseline_scores.append(mae)
    score=mae


    # adapt_model = ULSIF(RandomForestRegressor(n_estimators=100, max_depth=10),
    #      Xt=Xt, kernel="rbf",verbose=0,
              # lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.]) #random_state=0
    # adapt_model = RULSIF(RandomForestRegressor(n_estimators=100, max_depth=10), 
    #     Xt=Xt, kernel="rbf", alpha=0.1,verbose=0,
    #            lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], random_state=0)
    # adapt_model=NearestNeighborsWeighting(RandomForestRegressor(n_estimators=100, max_depth=10), 
    #     verbose=0,
    #     n_neighbors=5, Xt=Xt, random_state=0)
    # adapt_model= IWC(RandomForestRegressor(n_estimators=100, max_depth=10), 
    #     classifier=RandomForestRegressor(n_estimators=100, max_depth=10),
    #         Xt=Xt, random_state=0)
    # adapt_model=TrAdaBoostR2(RandomForestRegressor(n_estimators=100, max_depth=10),
    #     verbose=0,
    #  n_estimators=10, Xt=Xt.iloc[:10], yt=yt[:10], random_state=0)
    adapt_model=RegularTransferNN(src_only.task_, loss="mse", lambdas=1., random_state=0)

    # Fit the model.

    adapt_model.fit(Xt.iloc[:3], yt[:3],epochs=100, verbose=0)
    Xt=Xt.iloc[3:]
    yt=yt[3:]

    # Get the score on target data
    # adapt_score = adapt_model.score(Xt, yt) #R2
    # print("\tadapt_model score: ", adapt_score)
    # adapt_scores.append(adapt_score)
    # print(yt)
    
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
    # r2 = r2_score(yt, y_pred)
    # rmse = np.sqrt(mean_squared_error(yt, y_pred))
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


print(f"average improvement: {(average_improvement/(ncluster-fails))*100} %")
print(f"average worse: {(average_worse/(ncluster-fails))*100} %")
print(f"baseline avg. mae:{sum(baseline_scores)/len(baseline_scores)}")
print(f"adapt avg. mae:{sum(adapt_scores)/len(adapt_scores)}")
print(f"num_improvement:{num_improvement}")
print(f"num_worse:{num_worse}")

print("base std:",np.std(baseline_scores))
print("adapt std:",np.std(adapt_scores))
