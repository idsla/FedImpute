
<p align="center"><img src="docs/icon.jpg" width="400" height="240"></p>

# FedImpute: A Benchmarking and Evaluation Tool for Federated Imputation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

FedImpute is a benchmarking and evaluation tool to assess the effectiveness of federated imputation across various missing data scenarios. 

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
simulation_results = simulator.simulate_scenario(data, data_config, num_clients = 10, dp_strategy='iid-even', ms_mech_type='mcar')
```

### Step 3. Execute Federated Imputation Algorithms
```python
from fedimpute.execution_environment import FedImputeEnv
env = FedImputeEnv()
env.configuration(imputer = 'gain', fed_strategy='fedavg', fit_mode = 'fed')
env.setup(
    clients_train_data=simulation_results['clients_train_data'],
    clients_train_data_ms=simulation_results['clients_train_data_ms'],
    clients_test_data=simulation_results['clients_test_data'],
    clients_seeds=simulation_results['clients_seeds'],
    data_config=data_config,
)

env.run_fed_imputation()
```
### Step 4. Evaluate imputation outcomes
```python
```

## FedImputeBench - Benckmarking Analysis Using FedImpute

We use FedImpute to initialize a benchmarking analysis for federated imputation algorithms. The repo for FedImputeBench can be found [here](https://github.com/sitaomin1994/FedImputeBench)