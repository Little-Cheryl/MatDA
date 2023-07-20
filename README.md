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

2) Set up a virtual environment for the modnet model 


```
conda create -n modnet
conda activate modnet
pip install modnet
```


### How to run the codes

3) Set up a virtual environment for the ROOST model

```
mkdir generation_outputs
sh decode.sh
```

The generation is saved in ./generation_outputs.

```
sq2formula.py
```

The sequences are then conver to formulas and the formula results are saved to formulas.csv

