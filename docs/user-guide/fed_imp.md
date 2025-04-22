
# Executing Distributed Imputation Algorithms

The `FedImputeEnv` class is the `execution_environment` module's main class. 
It is used to configure the federated imputation environment and execute federated imputation algorithms.

## Overview and Basic Usage

Use needs to initialize the `FedImputeEnv` class and configure the environment using the `configuration` 
method - what imputer to use, what federated strategy to use, and what fitting mode to use. 
Then, use the `setup_from_simulator` method to set up the environment using the simulated data from `simulator` class, 
see [Scenario Simulation Section](../user-guide/scenario_simulation.md). 
Finally, use the `run_fed_imputation` method to execute the federated imputation algorithms.

```python
from fedimpute.execution_environment import FedImputeEnv

env = FedImputeEnv(debug_mode=False)
env.configuration(imputer = 'mice', fed_strategy='fedmice')
env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation()
```

Note that if you use cuda version of torch, remember to set environment variable for cuda deterministic behavior first
```bash
# bash (linux)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# powershell (windows)
$Env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

## Environment Configuration

The `env.configuration()` method is used to configure the environment. It takes the following arguments:

**Options**:

- **imputer** (str) - name of imputation algorithm to use. Options: `fed_mean`, `fed_em`, `fed_ice`, `fed_missforest`, `gain`, `miwae`
- **fed_strategy** (str) - name of federated strategy to use. Options: `fedavg`, `fedprox`, `scaffold`, `fedavg_ft`
- **fit_mode** (str) - name of fitting mode to use - federated imputation, local-only imputation or centralized imputation. Options: `fed`, `local`, `central`
- **save_dir_path** (str) - path to persist clients and server training process information (imputation models, imputed data etc.) for future use.

**Other Params**:

- **imputer_params** (Union[None, dict]) = None - parameters for imputer 
- **fed_strategy_params** (Union[None, dict]) = None - parameters for federated strategy
- **workflow_params** (Union[None, dict]) = None - parameters for workflow - 
`Workflow` class contains the logic for federated imputation workflow. It is associated with each `Imputer` class. 
- The built-in workflows are: `ice` - for ICE based imputation, `em` - for EM imputation, `jm` - for joint modeling based imputation such as VAE or GAN based imputation.



## Supported Federated Imputation Algorithms

Federated Imputation Algorithms:

|     Method     |     Type      |       Fed Strategy       |  Imputer (code)  | Workflow  | Reference                                                                                                                                                                                   |
|:--------------:|:-------------:|:------------------------:|:----------------:|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    Mean    |    Non-NN     |          `local`, `fedmean`             |    `mean`    | `MEAN`  | -                                                                                                                                                                                           |
|     EM     |    Non-NN     |         `local`, `fedem`             |     `em`     | `EM`      | [EM](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_EM.py), [FedEM](https://arxiv.org/abs/2108.10252)                                     |
|    MICE     |    Non-NN     |            `local`, `fedmice`             |    `mice`     | `ICE`     | [FedICE](https://pubmed.ncbi.nlm.nih.gov/33122624/)                                                                                                                                         |
| MissForest |    Non-NN     |            `local`, `fedtree`                    | `missforest` | `ICE`     | [MissForest](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_missforest.py), [Fed Randomforest](https://pubmed.ncbi.nlm.nih.gov/35139148/) |
|     MIWAE      |      NN       |    `local`, `fedavg`, ...    |     `miwae`      | `JM`      | [MIWAE](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_miwae.py)                                                                          |
|      GAIN      |      NN       | `local`, `fedavg`, ... |      `gain`      | `JM`      | [GAIN](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_gain.py)                                                                            |
|     Not-MIWAE      |      NN       |     `local`, `fedavg`, ...     |     `notmiwae`      | `JM` | [Not-MIWAE](https://arxiv.org/abs/2006.12871)
|     GNR      |      NN       |     `local`, `fedavg`, ...    |     `gnr`      | `JM` | [GNR](https://dl.acm.org/doi/abs/10.1145/3583780.3614835?casa_token=o8dv16sHJcMAAAAA:aAIvug_7cp9oUJSB7ZfTvzUksPyuP6Jbcl3TlHsvXXGEwIe4AbQuHCTlxXZtjDKlymfO30n2o-E9iw)

Federated Strategies:

|   Method   |      Type       | Fed_strategy(code) | Reference      |
|:----------:|:---------------:|:------------------:|:---------------|
|   Local   |    non-federated    |      `local`      | -     |
|   FedMean   |    traditional    |      `fedmean`      | -     |
|   FedEM   |    traditional    |      `fedem`      | [FedEM]()     |
|   FedMICE   |    traditional    |      `fedmice`      | [FedMICE]()     |
|   FedTree   |    traditional    |      `fedtree`      | [FedTree]()     |
|   FedAvg   |    global FL    |      `fedavg`      | [FedAvg](https://arxiv.org/pdf/1602.05629)     |
|  FedProx   |    global FL    |     `fedprox`      | [FedProx](https://arxiv.org/pdf/1812.06127)    |
|  Scaffold  |    global FL    |     `scaffold`     | [Scaffold](https://arxiv.org/pdf/1910.06378)   |
|  FedAdam   |    global FL    |     `fedadam`      | [FedAdam](https://arxiv.org/pdf/2003.00295)    |
| FedAdagrad |    global FL    |    `fedadagrad`    | [FedAdaGrad](https://arxiv.org/pdf/2003.00295) |
|  FedYogi   |    global FL    |     `fedyogi`      | [FedYogi](https://arxiv.org/pdf/2003.00295)    |
| FedAvg-FT  | personalized FL |    `fedavg_ft`     | [FedAvg-FT]()  |


## Environment Setup

After configuring environment, we need to initialize the environment -  initialize `Client`s, `Server` objects with simulated data from simulation module.
Currently, the `FedImputeEnv` class supports the two ways to set up the environment. First way is to directly setup the environment from `simulator` 
class by using `env.setup_from_simulator(simulator)` method. 
```python
env.setup_from_simulator(simulator, verbose=1)
```

The second way is to setup the environment by using `env.setup_from_data()` method. It can be used
in the scenario where user have their own data that not simulated from simulator class. Example:

```python
import numpy as np

clients_train_data = [np.random.rand(100, 10) for _ in range(10)]
clients_train_data_ms = [np.random.rand(100, 10) for _ in range(10)]
clients_test_data = [np.random.rand(100, 10) for _ in range(10)]
global_test = np.random.rand(100, 10)
data_config = {
    'target': 9,
    'task_type': 'regression',
    'clf_type': None,
    'num_cols': 9,
}
clients_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

env.setup_from_data(
    clients_train_data, clients_test_data, clients_train_data_ms, clients_seeds, global_test, data_config, verbose=1
)
```

## Execute Federated Imputation

After setting up the environment, we can execute the federated imputation algorithms using `run_fed_imputation()` method. Currently, we support two types of simulation execution (1) Run FL in sequential mode (`run_type="sequential"`), in this model, there is no parallel, whole processes of imputation for clients  run sequantially by using for loop (2) Run federated imputation in parallel mode (`run_type="parallel"`), it will simulate different processes for clients and server and then using workflow to manage communication between clients and server to approach the real world FL environment.

```python
env.run_fed_imputation(run_type='squential')
```

## Monitoring Imputation Process

We provide the Tensorboard utilty so that user can monitoring the imputation progress in real time. By using tensorboard, user need to run the following command in the terminal:

```{bash}
tensorboard --logdir .logs
```

We also provide another API `env.tracker.visualize_imputation_process()` it will show the line chart of imputation process measured by imputation quality or loss, it can only be run after imputation finished unlike tensorboard utility.

## Develop New Federated Imputation Methods

`FedImputeEnv` class supports functionality to register new imputers, strategies and workflows. User can develope their new imputation methods, federated strategies and workflow. Basically, these three components have to work tightly to formalize a federated imputation algorithm.

To develop new workflow, user need to implement a new workflow by inherit the `Workflow` class from `fedimpute.execution_environment.workflow`. In implementation of workflow, user need to think how to allow clients and server to interact with each other.

To develop new federated strategies, user need to implement new strategy for both client and server. 

To develop new imputers, user need to implement new imputer by inherit the one of two abstract class `BaseMLImputer` and `BaseNNImputer` from `fedimpute.execution_environment.imputation.base`, one for traditional methods and another for generative model based methods, and implement all its abstract methods (interfaces). Because, each imputer is associated with federated strategy and workflow, user need also to think how to make developed new imputer to be compatible with existed or new federated strategy and workflows.

Then user can use `env.register.register_imputer`, `env.register.register_strategy`, `env.register.register_workflow` to register these new developed classes. After registeration, user can use them in `FedImputeEnv` following the same way as using built-in methods. We provided a detailed example in tutorials.

## Miscellaneous

- **verbose** (int) - Verbosity level. 0: no output, 1: minimal output, 2: detailed output
- **seed** (int) - Seed for reproducibility
- **logging** (bool) - Whether to log the training process