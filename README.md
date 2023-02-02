# Reproducibility Study Label-Free XAI

## Summary
This repository includes code for implementations, experiments and supplementary studies used for reproducing the work and the experiments of the work [ICML 2022 paper](https://arxiv.org/abs/2203.01928): 'Label-Free Explainability for Unsupervised Models' by Jonathan Crabbé and Mihaela van der Schaar.



## 1. Installation
Make sure that you installed python 3.8. Then, from bash:
1. Create a python virtual environment with name env in the root folder of this repository:
```bash
python -m venv env
```
If you are using Conda, create virtual environment as follows:

First update the Conda:
```bash
conda update conda --all
```
Create a new environment:
```bash
conda create --name env python=3.8
```

2. Activate the python virtual environment:
```bash
source ./env/bin/activate 
```
If you are using Conda,
```bash
conda activate env
```

3. Upgrade pip:
```bash
pip install --upgrade pip
```

4. Install libraries required to execute our code: 
```bash
pip install -r requirements.txt
```

## 2. Reproducing the original paper results

### MNIST experiments
In the `experiments` folder, run the following script
```shell
python -m mnist --name experiment_name
```
where experiment_name can take the following values:

| experiment_name      | description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| consistency_features | Consistency check for label-free<br/> feature importance (authors' paper Section 4.1) |
| consistency_examples | Consistency check for label-free<br/> example importance (authors' paper Section 4.1) |
| roar_test            | ROAR test for label-free<br/> feature importance (authors' paper Appendix C)          |
| pretext              | Pretext task sensitivity<br/> use case (authors' paper Section 4.2)                   |
| disvae               | Challenging assumptions with <br/> disentangled VAEs (authors' paper Section 4.3)     |


The resulting plots and data are saved at the folder `results/mnist`.

### ECG5000 experiments
Run the following script
```shell
python -m ecg5000 --name experiment_name
```
where experiment_name can take the following values:

| experiment_name      | description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| consistency_features | Consistency check for label-free<br/> feature importance (authors' paper Section 4.1) |
| consistency_examples | Consistency check for label-free<br/> example importance (authors' paper Section 4.1) |

The resulting plots and data are saved `results/ecg5000`.

### CIFAR10 experiments
Run the following script
```shell
python -m cifar10
```
The experiment can be selected by changing the experiment_name
parameter in [this file](simclr_config.yaml).
The parameter can take the following values:

| experiment_name      | description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| consistency_features | Consistency check for label-free<br/> feature importance (authors' paper Section 4.1) |
| consistency_examples | Consistency check for label-free<br/> example importance (authors' paper Section 4.1) |



The resulting plots and data are saved at `results/cifar10`.

### dSprites experiment
Run the following script
```shell
python -m dsprites
```
The experiment needs several hours to run since several VAEs are trained.
The resulting plots and data are saved at `results/dsprites`.



## 3. Reproducing the additional experiments results

### Tiny ImageNet Experiments
In the `experiments` folder, run the following script
```shell
python -m imagenet --name experiment_name
```
where experiment_name can take the following values:

| experiment_name      | description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| consistency_features | Consistency check for label-free<br/> feature importance (authors' paper Section 4.1) |
| consistency_examples | Consistency check for label-free<br/> example importance (authors' paper Section 4.1) |


### CORA Experiment
In the `experiments` folder, run the following script
```shell
python -m cora --name consistency_features
```
### AGNews Experiment
In the `experiments` folder, run the following script
```shell
python -m agnews --name consistency_examples
```

### MNIST Experiments
#### Challenging the Generalizability of the authors' Assumptions on Disentangled VAEs
In the `experiments` folder, run the following scripts: 
```shell
python -m mnist --name disvae --n_runs 5 --reg_prior reg_param --attr_method_name method_name
```


| argument      | description                                                                  |
|----------------------|------------------------------------------------------------------------------|
| name | The name of the experiment to execute. In our case is `disvae` |
| n_runs | The number of runs for the experiment|
| batch_size  | The batch size to use for running the experiments |
| random_seed  | The random seed to use for the experiments |
| attr_method_name | What type of attribution method to use for the experiment. The value of the argument can be either `GradientShap` or `IntegratedGradients`|
| reg_prior | The  regularization attribution prior parameter to use. Note that with that being 0 or None no attribution prior will be used|
| load_models               | Whether to load models from files. The files must be given in the folders in which they were generated|
| load_metrics               | Whether to load metrics from files. The files must be given in the folders in which they were generated|

and reg_prior can take the following values:

| reg_prior   | 
|----------------------|
|  0.001 | 
| 0.005 | 
| 0.01 |
| 0.1 |


The resulting plots and data are saved at the folder `results/mnist/vae`.



## 4. Reproducing all the experiments results with a Jupyter notebook

To execute all the experiments in a notebook either from scratch or by loading pretrained models and data generated already from our experiments, please use the jupyter notebook named `run_experiments.ipynb` in the `experiments` folder.

## 4. Details of what the repository includes:
This code repository contains:
1. Implementation of LFXAI, a framework to explain the latent representations of unsupervised black-box `models` with the help of usual feature importance and example-based methods. It was introduced in the work of the authors of the Crabbé and van der Schaar.

2. Extensions/Additions to the LFXAI library:
    1. Added attr_priors.py file in `models` folder, that includes the total variation attribution prior penalty function
    2. Updated the VAE class of the images.py module in `models` folder to include support for using attribution priors
    3. Added a method attribute_auxiliary_single in features.py module of `explanations` folder that does the same thing as the method attribute_auxiliary but on a single batch of data.

3. Original Experiments Introduced by the authors:
    1. cifar10.py: Feature Importance, Example Importance
    2. dsprites.py: Disentangled VAEs Assumptions
    3. ecg5000.py: Feature Importance, Example Importance
    4. mnist.py: Feature Importance, Example Importance, Disentangled VAEs Assumptions, learned Pretext Task Representations experiments

4. Additional Experiments for reproducing the authors' work:
    1. agnews.py: Text explainability by example importance
    2. mnist.py: Experiments on Disentangled VAEs with attribution priors
    3. imagenet.py: Feature Importance, Example Importance
    4. cora.py: Graph explainability by feature importance
