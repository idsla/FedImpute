
import numpy as np
import pandas as pd
import tabulate
import random
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'

# %% [markdown]
# # Load Data
print("Loading data...")
from fedimpute.data_prep import load_data, display_data, column_check
from fedimpute.scenario import ScenarioBuilder
data, data_config = load_data("fed_heart_disease")
scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_real_scenario(
    data, data_config,
)
scenario_builder.summarize_scenario()
scenario_builder.visualize_missing_pattern(
    client_ids=[0, 1, 2, 3], data_type='train', fontsize=20, save_path='./plots/real_pattern_train.png'
)
scenario_builder.visualize_missing_pattern(
    client_ids=[0, 1, 2, 3], data_type='test', fontsize=20, save_path='./plots/real_pattern_test.png'
)

# %% [markdown]
# # Running Federated Imputation
from fedimpute.execution_environment import FedImputeEnv

print("Running federated imputation...")
env = FedImputeEnv(debug_mode=False)
env.configuration(imputer = 'mice', fed_strategy='fedmice', workflow_params = {'early_stopping_metric': 'loss'})
env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation(verbose=1)

# %% [markdown]
# # Evaluation
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()

X_train_imps, y_trains = env.get_data(client_ids='all', data_type = 'train_imp', include_y=True)
X_tests, y_tests = env.get_data(client_ids='all', data_type = 'test', include_y=True)
X_test_imps = env.get_data(client_ids='all', data_type = 'test_imp')
X_global_test, y_global_test = env.get_data(data_type = 'global_test', include_y = True)
X_global_test_imp = env.get_data(data_type = 'global_test_imp')
data_config = env.get_data(data_type = 'config')

# %% [markdown]
# ### Federated Prediction
print("Running federated regression analysis...")
evaluator.run_fed_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)
evaluator.show_fed_regression_results()
