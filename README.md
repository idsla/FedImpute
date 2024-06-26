# Complementarity Adjusted Federated (Cafe) Imputation

This is the code repo for paper - Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity

![cafe](./figures/cafe.png)


## Setup Environment and Data

Setup conda environment in terminal
```
conda env create -f environment.yml
conda activate fed_imp_new
```

Set up project structure

```
python setup_project.py
```

Download dataset to data folder
```
python scripts/download_data.py
```


## Federated Imputation

Run following command to simulate heterogenenous missing data FL scenarios and perform federated imputation with Cafe and baselines.

```
python run_fed_imputation.py
```

This script will run following the configuration in `conf/exp_config_imp.yaml` file, it loads the `conf/cnofig_vars/exp_demo.yaml` which includes configuration of data partition strategy, missing data simulation strategy and imputation methods, etc. and also for gain and miwae experiments.

By default, it uses multiprocessing to speed up, you can set `mtp` to `false` in config files to disable it. By default, it will run all scenarios `ideal`, `complex1` for 3 ice baselines (local ice, centralized ice, fed-ice), and cafe ice, you can change it in config to run specific scenarios `complex2`, `s1`, `s2`, `s3`, `s4` and baselines. 

To run experiemnts for gain and miwae, 3 gain baselines (local gain, centralized gain, fed-gain), 3 miwae baselines (local miwae, centralized miwae, fed-miwae), change the `config_tmpl` in `conf/exp_config_imp.yaml` to `imp_config_tmplate_gain` or `imp_config_tmplate_miwae` and change the option below `config_vars` to `exp_demo_gain` or `exp_demo_miwae`.

After finishing running, the results will be stored in json files under a folder of `\results\raw_results` folder. Run the following script to process resuls into a excel file stored in `\results\processed_results`. 

```
python scripts/results_fedimp.py
```

We give a `./notebook/demo.ipynb` to show the analysis of imputation results in a more readable format.

## Federated Prediction

Run following command to perform federated prediction of imputed datasets of each clients.

```
export CUBLAS_WORKSPACE_CONFIG=:4096:8
pythion run_fed_prediction.py --mtp True --num_processes 5
```

After finishing running, the results will be stored in json files under a folder of `\results\raw_results` folder. Run the following script to process resuls into a excel file stored in `\results\processed_results`. 

```
python scripts/results_fedpred.py
```
By default, we will use multi-processing to speed up, if you have low memory, you can set it to False.

We give a `./notebook/demo.ipynb` to show the analysis of prediction results in a more readable format.


