
# Executing Federated Imputation Algorithms

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

env = FedImputeEnv()
env.configuration(imputer = 'gain', fed_strategy='fedavg', fit_mode = 'fed')
env.setup_from_simulator(simulator = simulator, verbose=1)
env.run_fed_imputation()
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

|     Method     |     Type      |               Fed Strategy               |  Imputer (code)  | Workflow  | Reference                                                                                                                                                                                   |
|:--------------:|:-------------:|:----------------------------------------:|:----------------:|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    Fed-Mean    |    Non-NN     |                    -                     |    `fed_mean`    | `simple`  | -                                                                                                                                                                                           |
|     Fed-EM     |    Non-NN     |                    -                     |     `fed_em`     | `em`      | [EM](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_EM.py), [FedEM](https://arxiv.org/abs/2108.10252)                                     |
|    Fed-ICE     |    Non-NN     |                    -                     |    `fed_ice`     | `ice`     | [FedICE](https://pubmed.ncbi.nlm.nih.gov/33122624/)                                                                                                                                         |
| Fed-MissForest |    Non-NN     |                    -                     | `fed_missforest` | `ice`     | [MissForest](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_missforest.py), [Fed Randomforest](https://pubmed.ncbi.nlm.nih.gov/35139148/) |
|     MIWAE      |      NN       |     `fedavg`, `fedprox`, `fedavg_ft`     |     `miwae`      | `jm`      | [MIWAE](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_miwae.py)                                                                          |
|      GAIN      |      NN       |     `fedavg`, `fedprox`, `fedavg_ft`     |      `gain`      | `jm`      | [GAIN](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_gain.py)                                                                            |

Federated Strategies:

|   Method    |      Type       | Fed_strategy(code) | Reference     |
|:-----------:|:---------------:|:------------------:|:--------------|
|   FedAvg    |    global FL    |      `fedavg`      | [FedAvg]()    |
|   FedProx   |    global FL    |     `fedprox`      | [FedProx]()   |
|  Scaffold   |    global FL    |     `scaffold`     | [Scaffold]()  |
|  FedAvg-FT  | personalized FL |    `fedavg_ft`     | [FedAvg-FT]() |


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

After setting up the environment, we can execute the federated imputation algorithms using `run_fed_imputation()` method.

```python
env.run_fed_imputation()
```

## Miscellaneous

- **verbose** (int) - Verbosity level. 0: no output, 1: minimal output, 2: detailed output
- **seed** (int) - Seed for reproducibility
- **logging** (bool) - Whether to log the training process