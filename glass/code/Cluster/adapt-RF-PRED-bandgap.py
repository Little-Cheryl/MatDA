import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from adapt.instance_based import KMM
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from matminer.featurizers.conversions import StrToComposition
import os
from matminer.featurizers.composition import ElementProperty
from adapt.feature_based import PRED

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

y = df2['gap expt'].values
excluded = ["formula", "composition", "formula", "gap expt"]
X = df2.drop(excluded, axis=1)

with open(file_path, 'r') as file:
    cluster_ids = json.load(file)

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
    
    src_only = RandomForestRegressor(n_estimators=100, max_depth=10)
    print(f"{i}\t",end="")
    src_only.fit(Xs, ys)

    y_pred = src_only.predict(Xt)
    mae = mean_absolute_error(yt, y_pred)
    print("src_only mae: ", mae, end="")
    baseline_scores.append(mae)
    score=mae

    adapt_model = PRED(
        estimator=RandomForestRegressor(n_estimators=100, max_depth=10),
        Xt=Xt[:3], yt=yt[:3], copy=True, pretrain=True, verbose=1, random_state=None
    )

    adapt_model.fit(Xs, ys)
    Xt=Xt.iloc[3:]
    yt=yt[3:]

    y_pred = adapt_model.predict(Xt)

    nan_count = np.sum(np.isnan(y_pred))
    if nan_count !=0:
        print("\tadapt_model mae: ", 'NaN', end="")
        fails+=1
        continue

    mae = mean_absolute_error(yt, y_pred)
    print("\tadapt_model mae: ", mae, "\n", end="")
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

print(f"average improvement: {(average_improvement/ncluster)*100} %")
print(f"average worse: {(average_worse/ncluster)*100} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print(f"adapt avg. accuracy:{sum(adapt_scores)/len(adapt_scores)}")
print(f"num_improvement:{num_improvement}")
print(f"num_worse:{num_worse}")
print("base std:",np.std(baseline_scores))
print("adapt std:",np.std(adapt_scores))

