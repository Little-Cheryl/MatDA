## MatDA
Code for Improving realistic material property prediction using domain adaptation based machine learning.


### Installation


1) Set up a virtual environment for adapting models

~~~
conda create -n adapt
conda activate adapt
pip install adapt
pip install matminer
pip install numpy==1.23
~~~

2) Set up a virtual environment for the modnet model as [[Modnet]](ttps://github.com/ppdebreuck/modnet)


```
conda create -n modnet python=3.9
conda activate modnet
pip install modnet
```


### How to run the codes

1) As mentioned in the paper, we have 5 bandgap datasets for regression and 5 glass datasets for classification.
2) For regression problems, all codes and data are under the bandgap folder, and identical to the glass folder.
3) We separate all codes into Single and Cluster based on the dataset they use. Codes under the Single folder can be used to evaluate SparseXSingle and SparseYSingle datasets, and codes under the Cluster folder are for LOCO, SparseXCluster, and SparseYCluster.
4) Here we take the bandgap-SparseXSingle as an example:

```
cd ./bandgap/code/Single/
python adapt-RF-KMM.py
```
Users can choose the algorithms they want to evaluate, and the dataset can also be modified in each code file.

