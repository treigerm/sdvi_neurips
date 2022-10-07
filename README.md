# Code for "Rethinking Variational Inference for Probabilistic Programs with Stochastic Support"

## Installation and setup

This repository is using [Poetry](https://python-poetry.org/) to manage dependencies. 
To run code you therefore first have to install Poetry on your system.

Then to install all the dependencies run
```bash
poetry install
```
and then you should be good to go.

## Outline of repository

- `data`
    - `data/airline` contains the data for the GP experiment
    - `data/gmm` contains the data for the GMM experiment
- `models` contains the implementations of the Pyro models
    - `models/pyro_extensions/` contains our implementation of SDVI as well as our Pyro implementation of DCC
- `notebooks/paper_figures.ipynb` contains the code to reproduce the figures in the paper 
- This repository uses the [Hydra](https://hydra.cc/) configuration management systems to configure experiments. There are several configuration directories for the different methods.
    - `conf_pyro_extension` contains the configurations for SDVI
    - `baselines_config` contains the configurations for the baselines for the experiment in Section 6.1
    - `gmm_baselines_conf` contains the configurations for the baselines for the GMM experiment
    - `gp_baselines_conf` contains the configurations for the baselines for the GP experiment

For the models that require access to data stored on disk require an absolute path in their configuration file. 
For anonymization these absolute paths have been removed and you will manually have to enter these paths for your local machine.
The necessary places in the configuration files are marked with `TODO`s.


## Running experiments

### Figure 1

The code for the experiments of Figure 1 is in `scripts/make_motivating_example_plot.py` which can be run using
```
poetry run python scripts/make_motivating_example_plot.py $ID
```
where `$ID` is an integer identifying this run. 
One run creates 20 replications for each method. 
So to reproduce Figure 1 run this script 5 times with different IDs and then the code to combine the results and create the final figure is in the notebook `notebooks/paper_figures.ipynb`.

### Program with Normal Distributions

Running SDVI for this model can be done using 
```
poetry run python run_exp_pyro_extension.py \
    name=normal_model_sdvi \
    sdvi.forward_kl_iter=1000 \
    sdvi.forward_kl_num_particles=100 \
    sdvi.elbo_estimate_num_particles=1000 \
    sdvi.exclusive_kl_num_particles=5 \
    model=normal_model \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=100000
```

The baselines can be run using the `run_baselines.py` script e.g. to run the BBVI baseline:
```
poetry run python run_baselines.py \
    name=normal_model_bbvi \
    inference_algo=autonormal_svi \
    inference_algo.gradient_estimator=score \
    inference_algo.num_particles=10 \
    inference_algo.num_steps=10000
```
The `baselines_config` directory lays out the other configurations for the other baselines.

### Infinite Gaussian Mixture Model

Running SDVI for this model can be done using
```
poetry run python run_exp_pyro_extension.py \
    name=infinite_gmm_d100_sdvi \
    sdvi.exclusive_kl_num_particles=10 \
    sdvi.elbo_estimate_num_particles=100 \
    model=infinite_gmm_d100 \
    posterior_predictive_num_samples=10 \
    sdvi.learning_rate=0.1 \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=20000
```

The baselines can be run using the `run_gmm_baselines.py` script e.g. to run the BBVI baseline:
```
poetry run python run_gmm_baselines.py \
    name=infinite_gmm_d100_bbvi \
    inference_algo=bbvi_finite \
    inference_algo.evaluate_every_n=100 \
    inference_algo.num_elbo_particles=10 \
    inference_algo.num_iterations=20000 \
    inference_algo.num_posterior_samples=10 \
    inference_algo.learning_rate=0.01 \
    inference_algo.upper_bound=25
```
The `gmm_baselines_conf` directory lays out the other configurations for the other baselines.

### Inferring Gaussian Process Kernels

Running SDVI for this model can be done using
```
poetry run python run_exp_pyro_extension.py \
    name=gp_grammar_sdvi \
    sdvi.exclusive_kl_num_particles=1 \
    sdvi.elbo_estimate_num_particles=100 \
    model=gp_kernel_learning \
    posterior_predictive_num_samples=10 \
    sdvi.learning_rate=0.005 \
    sdvi.save_metrics_every_n=200 \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=1000000
```

The baselines can be run using the `run_gp_baselines.py` script e.g. to run the BBVI baseline:
```
poetry run python run_gp_baselines.py \
    name=gp_grammar_bbvi \
    inference_algo.learning_rate=0.005 \
    inference_algo.num_elbo_particles=10 \
    inference_algo.num_posterior_samples=10 \
    inference_algo.num_iterations=100000 \
    inference_algo.evaluate_every_n=100
```
The `gp_baselines_conf` directory lays out the other configurations for the other baselines.