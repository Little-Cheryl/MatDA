## MatDA
Code for Improving realistic material property prediction using domain adaptation based machine learning.


### Installation


1) Set up a virtual environment for adapting models

~~~Conda Setup:
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb 
~~~

2) Set up a virtual environment for the modnet model


```
cd code/Diffusion-LM/improved-diffusion 
mkdir diffusion_models
sh run.sh
```

the trained model is saved in ./diffusion_models

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

