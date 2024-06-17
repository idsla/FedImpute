
# Installation
---
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

# Basic Usage
---
### Step 1. Prepare Data
```python
import numpy as np
data = np.random.rand(10000, 10)
data_config = {
    'target': 9,
    'task_type': 'regression',
    'clf_type': None,
    'num_cols': 10,
}
```

### Step 2. Simulate Federated Missing Data Scenario
```python
from fedimpute.simulator import Simulator
simulator = Simulator()

# classifical simulation function
simulation_results = simulator.simulate_scenario(
    data, data_config, num_clients = 10, dp_strategy='iid-even', ms_mech_type='mcar', verbose=1
)

# lite simulation function
simulation_results = simulator.simulate_scenario_lite(
    data, data_config, num_clients = 10, dp_strategy='niid-dir@0.1', ms_scenario = 'mar-heter', verbose=1
)
```

### Step 3. Execute Federated Imputation Algorithms
```python
from fedimpute.execution_environment import FedImputeEnv
env = FedImputeEnv()
env.configuration(imputer = 'gain', fed_strategy='fedavg', fit_mode = 'fed')
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