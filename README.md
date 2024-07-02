<p align="center"><img src="docs/icon.jpg" width="400" height="240"></p>
<h2 align='center'> FedImpute: a benchmarking and evaluation tool for federated imputation across various missing data scenarios. </h2>

<div align="center">
    
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://idsla.github.io/FedImpute/)

</div>

FedImpute is a benchmarking tool for the evaluation of federated imputation algorithms over various missing data scenarios under horizontally partitioned data.

- **Documentation:** [Documentation](https://idsla.github.io/FedImpute/)
- **Paper:** [FedImpute]()
- **Source Code:** [Source Code](https://github.com/idsla/FedImpute/)
- **Benchmarking Analysis:** [FedImputeBench](https://github.com/idsla/FedImpute/tree/fedimpute_bench)

## Installation
Install python >= 3.8.0
```bash
python -m venv ./venv

# window gitbash
source ./venv/Scripts/activate

# linux/unix
source ./venv/bin/activate
```
Install the required packages
```bash
pip install -r requirements.txt
```
## Basic Usage

### Step 1. Prepare Data
```python
import numpy as np
data = np.random.rand(10000, 10)
data_config = {
    'task_type': 'regression',
    'num_cols': 9,
}
```

### Step 2. Simulate Federated Missing Data Scenario
```python
from fedimpute.simulator import Simulator
simulator = Simulator()
simulation_results = simulator.simulate_scenario(
    data, data_config, num_clients = 10, dp_strategy='iid-even', ms_mech_type='mcar', verbose=1
)
```

### Step 3. Execute Federated Imputation Algorithms
```python
from fedimpute.execution_environment import FedImputeEnv
env = FedImputeEnv()
env.configuration(imputer = 'fed_ice', fed_strategy='fedavg', fit_mode = 'fed')
env.setup_from_simulator(simulator = simulator, verbose=1)

env.run_fed_imputation()
```
### Step 4. Evaluate imputation outcomes
```python
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
evaluator.evaluate(env, ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed'])
evaluator.show_results()
```

## Supported Federated Imputation Algorithms

Federated Imputation Algorithms:

|     Method     |     Type      |               Fed Strategy               |  Imputer (code)  | Reference                                                                                                                                                                                   |
|:--------------:|:-------------:|:----------------------------------------:|:----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    Fed-Mean    |    Non-NN     |                    -                     |    `fed_mean`    | -                                                                                                                                                                                           |
|     Fed-EM     |    Non-NN     |                    -                     |     `fed_em`     | [EM](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_EM.py), [FedEM](https://arxiv.org/abs/2108.10252)                                     |
|    Fed-ICE     |    Non-NN     |                    -                     |    `fed_ice`     | [FedICE](https://pubmed.ncbi.nlm.nih.gov/33122624/)                                                                                                                                         |
| Fed-MissForest |    Non-NN     |                    -                     | `fed_missforest` | [MissForest](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_missforest.py), [Fed Randomforest](https://pubmed.ncbi.nlm.nih.gov/35139148/) |
|     MIWAE      |      NN       |     `fedavg`, `fedprox`, `fedavg_ft`     |     `miwae`      | [MIWAE](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_miwae.py)                                                                          |
|      GAIN      |      NN       |     `fedavg`, `fedprox`, `fedavg_ft`     |      `gain`      | [GAIN](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_gain.py)                                                                            |

Federated Strategies:

|   Method    |      Type       | Fed_strategy(code) | Reference     |
|:-----------:|:---------------:|:------------------:|:--------------|
|   FedAvg    |    global FL    |      `fedavg`      | [FedAvg]()    |
|   FedProx   |    global FL    |     `fedprox`      | [FedProx]()   |
|  Scaffold   |    global FL    |     `scaffold`     | [Scaffold]()  |
|  FedAvg-FT  | personalized FL |    `fedavg_ft`     | [FedAvg-FT]() |



## FedImputeBench - Benckmarking Analysis Using FedImpute

We use `FedImpute` to initialize a benchmarking analysis for federated imputation algorithms. The repo for **FedImputeBench** can be found [here](https://github.com/sitaomin1994/FedImputeBench)

## Contact
For any questions, please contact [Sitao Min](mailto:sm2370@rutgers.edu)
