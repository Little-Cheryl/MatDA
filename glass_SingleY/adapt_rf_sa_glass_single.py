# -*- coding: utf-8 -*-
"""adapt-RF-SA-glass-single.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VUhnyKTS5AaV3xY7t2dgHQrWYN3AHZRU
"""

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Import KMM method form adapt.instance_based module
from adapt.instance_based import KMM
import json

file_path = 'glass_target_clusters50.json'
ncluster=50
featurefile = 'glass_features.csv'
property_column='gfa'

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

y = df2[property_column].values
excluded = ["formula", "composition", "formula", property_column]
X = df2.drop(excluded, axis=1)
with open(file_path, 'r') as file:
    cluster_ids = json.load(file)

from adapt.feature_based import SA
from sklearn.metrics import balanced_accuracy_score

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
    yt = df2["gfa"].iloc[target_id_list].values

    Xs = X.drop(target_id_list)
    ys = df2["gfa"].drop(target_id_list).values

    src_only = RandomForestClassifier(n_estimators=100, max_depth=10)
    print(f"{i}\t",end="")
    src_only.fit(Xs, ys)


    y_pred = src_only.predict(Xt)
    score = balanced_accuracy_score(yt, y_pred)

    print("src_only score: ", score, end="")

    baseline_scores.append(score)


    adapt_model= SA(estimator = RandomForestClassifier(n_estimators=100, max_depth=10),
       Xt=Xt, n_components=None, copy=True, verbose=1, random_state=None
    )
    # Fit the model.
    adapt_model.fit(Xs, ys)

    # Xt = Xt.iloc[3:]
    # yt = yt[3:]
    y_pred = adapt_model.predict(Xt)

    # Get the score on target data
    adapt_score = balanced_accuracy_score(yt, y_pred) #R2
    print("\tadapt_model score: ", adapt_score)
    adapt_scores.append(adapt_score)


    if adapt_score == score:
        num_improvement += 0
        average_improvement += 0
    elif adapt_score > score:
        num_improvement += 1
        # w = 1-(adapt_score/score)
        average_improvement += 1
    else:
        num_worse += 1
        # w = 1-(adapt_score/score)
        average_worse += -1

print(f"average improvement: {(average_improvement/(ncluster-fails))} %")
print(f"average worse: {(average_worse/(ncluster-fails))} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print(f"adapt avg. accuracy:{sum(adapt_scores)/len(adapt_scores)}")
print(f"num_improvement:{num_improvement}")
print(f"num_worse:{num_worse}")
print("base std:",np.std(baseline_scores))
print("adapt std:",np.std(adapt_scores))
