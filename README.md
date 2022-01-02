# Deep-Learning-Project-2021

## Prepare dependencies

### Load modules in Euler

```
env2lmod
module load gcc/8.2.0
module load python_gpu/3.8.5
module load cuda/11.3.1
```

### Set up a Python virtual environment (venv)

```
python -m venv venv
source venv/bin/activate
```

### Install modules

Install the dependencies given in `requirements.txt`. 

```
pip install -r requirements.txt
```

Nvidia Apex is not published in PyPI and has to be installed by following the instructions [here](https://github.com/NVIDIA/apex).

## Train models

Change directory to either of {resnet, vit, DemystifyLocalViT} and execute:

```
./run_train.sh gpu
```

## Generate combined and background datasets

```
cd combined_dataset
combined_dataset/run.sh
```

```
cd background_dataset
background_dataset/run.sh
```

## Test models

Change directory to either of {resnet, vit, DemystifyLocalViT} and execute:

```
./run_eval.sh gpu
```