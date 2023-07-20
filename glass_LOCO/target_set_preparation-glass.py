#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


# In[2]:


import json
file_path = 'glass_target_clusters50.json'
n_clusters=50
df = pd.read_csv("glass_features.csv")

df


# In[3]:


y = df['gfa'].values
excluded = ["formula",
              "composition", "formula", "gfa"]

X = df.drop(excluded, axis=1)


# In[4]:


kmeans = KMeans(n_clusters=n_clusters,max_iter=500,random_state=222)

# Fit the data to the KMeans model
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Get the cluster centers
centers = kmeans.cluster_centers_

# Print the cluster labels and centers
print("Cluster Labels:", labels)
print("Cluster Centers:", centers)


# In[5]:


cluster = {}
for i, x in enumerate(labels):
    if x in cluster:
        cluster[int(x)].append(i)
    else:
        cluster[int(x)] = [i]
for key in sorted(cluster):
    print(key, len(cluster[key]))


# In[6]:


# Save the dictionary to a file
with open(file_path, 'w') as file:
    json.dump(cluster, file)


# In[7]:


for index in cluster[0]:
    print(df.iloc[index]["formula"])


# In[ ]:




