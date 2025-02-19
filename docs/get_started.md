
# Installation
---
Firstly, install python >= 3.10.0, we have two ways to install

Install from pip:
```bash
pip install fedimpute
```

Install from package repo:
```bash
git clone https://github.com/idsla/FedImpute
cd FedImpute

# create virtual env
python -m venv ./venv

# window gitbash
source ./venv/Scripts/activate

# linux/unix
source ./venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

# Basic Usage
---
### Step 1. Prepare Data
```python
from fedimpute.data_prep import load_data, display_data

data, data_config = load_data("codrna")
print("Data Dimensions: ", data.shape)
print("Data Config:\n", data_config)
# Structure of data_config: 
#   {
#       'target': 'y', 
#       'task_type': 'classification', 
#       'natural_partition': False
#   }
data.head()
```

### Step 2. Build Distributed Missing Data Scenario
```python
from fedimpute.scenario import ScenarioBuilder

scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_simulated_scenario(
    data, data_config, num_clients = 4, dp_strategy='iid-even', ms_scenario='mnar-heter'
)
scenario_builder.summarize_scenario()
```

### Step 3. Execute Distributed Imputation Algorithms
Note that if you use cuda version of torch, remember to set environment variable for cuda deterministic behavior first

```bash
# bash (linux)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# powershell (windows)
$Env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

```python
from fedimpute.execution_environment import FedImputeEnv

env = FedImputeEnv(debug_mode=False)
env.configuration(imputer = 'mice', fed_strategy='fedmice')
env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation()
```


### Step 4. Evaluate imputation outcomes
```python
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
ret = evaluator.evaluate_all(
    env, metrics = ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed']
)
evaluator.show_results_all()
```