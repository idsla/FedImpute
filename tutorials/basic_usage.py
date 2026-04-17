
# %%
import numpy as np
import pandas as pd
import tabulate
import matplotlib.pyplot as plt
import random
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'


# %% [markdown]
# # Load Data
print("Loading data...")
from fedimpute.data_prep import load_data, display_data
data, data_config = load_data("codrna")
display_data(data)
print("Data Dimensions: ", data.shape)
print("Data Config:\n", data_config)

print(data.mean(numeric_only=True).round(8))
# validate data and data config
assert isinstance(data, pd.DataFrame)
assert data.shape == (5000, 9)
expected_columns = [f"X{i}" for i in range(1, 9)] + ["y"]
assert data.columns.tolist() == expected_columns
assert not data.isna().any().any()
assert np.isfinite(data.to_numpy()).all()
assert set(data["y"].unique()).issubset({0.0, 1.0})
assert data_config['target'] == 'y'
assert data_config['task_type'] == 'classification'
assert data_config['natural_partition'] == False
assert data_config['num_cols'] == 8

# %% [markdown]
# # Scenario Simulation
# ## Basic Usage
print("Creating scenario...")
from fedimpute.scenario import ScenarioBuilder

scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_simulated_scenario(
    data, data_config, num_clients = 4, dp_strategy='iid-even', 
    ms_scenario='mnar-heter'
)
print('Results Structure (Dict Keys):')
print(list(scenario_data.keys()))
scenario_builder.summarize_scenario()

# %% [markdown]
# # Running Federated Imputation
print("Running federated imputation...")
# %% [markdown]
# ## Basic Usage

# %%
from fedimpute.execution_environment import FedImputeEnv

env = FedImputeEnv(debug_mode=False)
env.configuration(imputer = 'mice', fed_strategy='fedmice')
env.setup_from_scenario_builder(
    scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation(verbose=2)

# %% [markdown]
# # Evaluation

# %% [markdown]
# ### Imputation Quality
from fedimpute.evaluation import Evaluator

X_trains = env.get_data(client_ids='all', data_type = 'train')
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
X_train_masks = env.get_data(client_ids='all', data_type = 'train_mask')

from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
ret = evaluator.evaluate_imp_quality(
    X_train_imps = X_train_imps,
    X_train_origins = X_trains,
    X_train_masks = X_train_masks,
    metrics = ['rmse', 'nrmse', 'sliced-ws']
)
evaluator.show_imp_results()

# %% [markdown]
# ### Get Imputed Data
# %%
X_train_imps, y_trains = env.get_data(client_ids='all', data_type = 'train_imp', include_y=True)
X_tests, y_tests = env.get_data(client_ids='all', data_type = 'test', include_y=True)
X_global_test, y_global_test = env.get_data(data_type = 'global_test', include_y = True)
data_config = env.get_data(data_type = 'config')

# %% [markdown]
# # regression analysis
print("Running local regression analysis...")
# %%
X_trains, y_trains = env.get_data(client_ids='all', data_type = 'train', include_y=True)
data_config = env.get_data(data_type = 'config')
ret = evaluator.run_local_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)

evaluator.show_local_regression_results(client_idx = 0)

# %% [markdown]
# ### Local Prediction
print("Running local prediction...")
ret = evaluator.run_local_prediction(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    X_tests = X_tests,
    y_tests = y_tests,
    data_config = data_config,
    model = 'lr',
    seed= 0
)
evaluator.show_local_prediction_results()

# %% [markdown]
# ### Federated Prediction

# %%
print("Running federated prediction...")
ret = evaluator.run_fed_prediction(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    X_tests = X_tests,
    y_tests = y_tests,
    X_test_global = X_global_test,
    y_test_global = y_global_test,
    data_config = data_config,
    model_name = 'lr',
    seed= 0
)

evaluator.show_fed_prediction_results()

# %% [markdown]
# # federated regression analysis
print("Running federated regression analysis...")
X_trains, y_trains = env.get_data(client_ids='all', data_type = 'train', include_y=True)
data_config = env.get_data(data_type = 'config')

evaluator.run_fed_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)
evaluator.show_fed_regression_results()