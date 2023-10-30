import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,balanced_accuracy_score
from matminer.featurizers.conversions import StrToComposition
import os
from matminer.featurizers.composition import ElementProperty
from adapt.feature_based import TCA
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

file_path = '../../data/SparseXSingle.json'
ncluster=50
featurefile = '../../data/glass_features.csv'


if not os.path.exists(featurefile):
    df1 = StrToComposition().featurize_dataframe(df, "formula")
    df1.head()
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df2 = ep_feat.featurize_dataframe(df1, col_id="composition")
    df2.head()
    excluded = ["formula", "composition", "formula", "gfa"]
    df2.to_csv(featurefile)
else:  #load the features
    df2 = pd.read_csv(featurefile)

y = df2['gfa'].values
excluded = ["formula", "composition", "formula", "gfa"]
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
    yt = df2["gfa"].iloc[target_id_list].values

    Xs = X.drop(target_id_list)
    ys = df2["gfa"].drop(target_id_list).values
    
    src_only = RandomForestClassifier(n_estimators=100, max_depth=10)
    print(f"{i}\t",end="")
    src_only.fit(Xs, ys)

    y_pred = src_only.predict(Xt)
    score = balanced_accuracy_score(yt, y_pred)
    print("src_only mae: ", score, end="")
    baseline_scores.append(score)

    adapt_model= TCA(estimator = RandomForestClassifier(n_estimators=100, max_depth=10),
       Xt=Xt, n_components=20, mu=0.1, kernel='rbf', copy=True, verbose=1, random_state=None
    )

    adapt_model.fit(Xs, ys)
    y_pred = adapt_model.predict(Xt)

    adapt_score = balanced_accuracy_score(yt, y_pred) 
    print("\tadapt_model score: ", adapt_score)
    adapt_scores.append(adapt_score)

    nan_count = np.sum(np.isnan(y_pred))
    if nan_count !=0:
        print("\tadapt_model mae: ", 'NaN', end="")
        fails+=1
        continue

    if adapt_score == score:
        num_improvement += 0
        average_improvement += 0
    elif adapt_score > score:
        num_improvement += 1
        average_improvement += 1
    else:
        num_worse += 1
        average_worse += -1

print(f"average improvement: {(average_improvement/(ncluster-fails))} %")
print(f"average worse: {(average_worse/(ncluster-fails))} %")
print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print(f"adapt avg. accuracy:{sum(adapt_scores)/len(adapt_scores)}")
print(f"num_improvement:{num_improvement}")
print(f"num_worse:{num_worse}")
print("base std:",np.std(baseline_scores))
print("adapt std:",np.std(adapt_scores))
