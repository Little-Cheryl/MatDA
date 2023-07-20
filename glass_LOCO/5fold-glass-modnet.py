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
from matminer.featurizers.conversions import StrToComposition
import modnet
from sklearn.model_selection import KFold

def convert_to_boolean(lst):
    boolean_list = [bool(element) for element in lst]
    return boolean_list


featurefile = 'glass_features.csv'
property_column='gfa'

df2 = pd.read_csv(featurefile)
y = df2[property_column].values
excluded = ["formula", "composition", "formula", property_column]
X = df2.drop(excluded, axis=1)


df = df2[["composition",property_column]]
df['composition']=df['composition'].map(Composition)

baseline_scores=[]

# from sklearn.model_selection import train_test_split
# split = train_test_split(range(100), test_size=0.1, random_state=1234)

data = MODData(
    materials=df["composition"], # you can provide composition objects to MODData
    targets=df["gfa"], 
    target_names=["gfa"],
    # num_classes = {'gfa':2}
)
data.featurize()
data.feature_selection(n=-1)
# from sklearn.model_selection import train_test_split
# split = train_test_split(range(100), test_size=0.1, random_state=1234)
# train, test = data.split(split)


from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kfold.split(X):
    # print("Train index:", train_index)
    # print("Test index:", test_index)
    train,test = data.split([train_index,test_index])


    model = MODNetModel([[['gfa']]],
                    weights={'gfa':1},
                    num_neurons = [[128], [64], [16], []],
                    n_feat = 150,
                    act =  "elu",
                    num_classes = {'gfa':2}
                   )

    model.fit(train,
          val_fraction = 0.1,
          lr = 0.002,
          batch_size = 64,
          # "num_classes": {'gfa':2},
          loss = 'binary_crossentropy',
          epochs = 200,
          verbose = 0,
          # num_classes = {'gfa':2}
         )

    pred = model.predict(test)
    y_pred = convert_to_boolean(pred.values)
    score = balanced_accuracy_score(test.df_targets.values,y_pred)

    print("src_only score: ", score)
    baseline_scores.append(score)
    print("all scores", baseline_scores)

print(f"baseline avg. accuracy:{sum(baseline_scores)/len(baseline_scores)}")
print("base std:", np.std(baseline_scores))




