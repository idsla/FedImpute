
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

LOCAL_PREDICTION_RTOL = 1e-6
LOCAL_PREDICTION_ATOL = 1e-8
PREDICTION_RTOL = 2e-2
PREDICTION_ATOL = 1e-2


####################################################################################
# %% [markdown]
# # Load Data
print("Loading data...")
from fedimpute.data_prep import load_data, display_data
data, data_config = load_data("codrna")
display_data(data)
print("Data Dimensions: ", data.shape)
print("Data Config:\n", data_config)

# validate data and data config
assert isinstance(data, pd.DataFrame)
assert data.shape == (5000, 9)
expected_columns = [f"X{i}" for i in range(1, 9)] + ["y"]
assert data.columns.tolist() == expected_columns
assert not data.isna().any().any()
assert np.isfinite(data.to_numpy()).all()
assert set(data["y"].unique()).issubset({0.0, 1.0, np.float64(0.0), np.float64(1.0), np.float64(0.9999999999999999)})
expected_means = pd.Series({
    "X1": 0.783610,
    "X2": 0.313509,
    "X3": 0.305492,
    "X4": 0.380526,
    "X5": 0.645553,
    "X6": 0.313939,
    "X7": 0.373722,
    "X8": 0.648571,
    "y": 0.324200
})
actual_means = data.mean(numeric_only=True)
pd.testing.assert_series_equal(
    actual_means,
    expected_means,
    check_names=False,
    check_dtype=False,
    rtol=1e-5,
    atol=1e-8,
)
assert data_config['target'] == 'y'
assert data_config['task_type'] == 'classification'
assert data_config['natural_partition'] == False
assert data_config['num_cols'] == 8
print("Passed all data validation checks!")

######################################################################################
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

# validate scenario_data structure and content
assert isinstance(scenario_data, dict)
expected_keys = {'clients_train_data', 'clients_test_data', 'clients_train_data_ms', 'clients_seeds', 'global_test_data', 'data_config', 'stats'}
assert set(scenario_data.keys()) == expected_keys
assert len(scenario_data['clients_train_data']) == 4
assert len(scenario_data['clients_test_data']) == 4
assert len(scenario_data['clients_train_data_ms']) == 4
assert len(scenario_data['clients_seeds']) == 4
assert isinstance(scenario_data['global_test_data'], pd.DataFrame)
assert scenario_data['data_config'] == data_config
expected_seeds = [6077, 577, 7231, 5504]
expected_missing_ratios = [0.47044444, 0.50944444, 0.46244444, 0.47044444]
expected_train_means = [0.45426866, 0.45586764, 0.45506127, 0.45423325]
expected_test_means = [0.45292317, 0.45825866, 0.44715748, 0.45541227]
expected_train_ms_means = [0.4311517, 0.46201121, 0.50622139, 0.52351541]
for i in range(4):
    # train data
    assert isinstance(scenario_data['clients_train_data'][i], pd.DataFrame)
    assert scenario_data['clients_train_data'][i].shape == (1125, 9)
    # test data
    assert np.isclose(np.mean(scenario_data['clients_train_data'][i]), expected_train_means[i], rtol=1e-8)
    assert isinstance(scenario_data['clients_test_data'][i], pd.DataFrame)
    assert scenario_data['clients_test_data'][i].shape == (113, 9)
    assert np.isclose(np.mean(scenario_data['clients_test_data'][i]), expected_test_means[i], rtol=1e-8)
    # train data with missing
    assert isinstance(scenario_data['clients_train_data_ms'][i], pd.DataFrame)
    assert scenario_data['clients_train_data_ms'][i].shape == (1125, 8)
    assert np.isclose(np.isnan(scenario_data['clients_train_data_ms'][i]).mean().mean(), expected_missing_ratios[i], rtol=1e-8)
    assert np.isclose(np.nanmean(scenario_data['clients_train_data_ms'][i]), expected_train_ms_means[i], rtol=1e-8)
    # seeds
    assert scenario_data['clients_seeds'][i] == expected_seeds[i] 
    
print("Passed all scenario simulation validation checks!")


#################################################################################
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

assert env.workflow.name == 'ICE (Imputation via Chain Equation)'
for client_idx in range(4):
    assert env.clients[client_idx].imputer.name == 'mice'
    assert env.clients[client_idx].fed_strategy.name == 'fedmice'
assert env.server.fed_strategy.name == 'fedmice'
print("Passed all federated imputation validation checks!")


################################################################################
# %% [markdown]
# # Evaluation

# %% [markdown]
# ### Imputation Quality
from fedimpute.evaluation import Evaluator

X_trains = env.get_data(client_ids='all', data_type = 'train')
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
X_train_masks = env.get_data(client_ids='all', data_type = 'train_mask')

# validate imputed data structure and content
expected_means = [
    (0.47049669, 0.45157754), (0.47229554, 0.44379585), 
    (0.47138838, 0.48238528), (0.47045685, 0.47645606)]
for i in range(4):
    assert X_trains[i].shape == (1125, 8)
    assert X_train_imps[i].shape == (1125, 8)
    assert X_train_masks[i].shape == (1125, 8)
    assert X_trains[i].isna().any().any() == False
    assert X_train_imps[i].isna().any().any() == False
    assert np.isclose(np.nanmean(X_trains[i]), expected_means[i][0], rtol=1e-6)
    assert np.isclose(np.nanmean(X_train_imps[i]), expected_means[i][1], rtol=1e-6)
print("Passed all imputed data validation checks!")

from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
ret = evaluator.evaluate_imp_quality(
    X_train_imps = X_train_imps,
    X_train_origins = X_trains,
    X_train_masks = X_train_masks,
    metrics = ['rmse', 'nrmse', 'sliced-ws']
)
evaluator.show_imp_results()

expected_results = {
    'rmse': [0.18124601, 0.18765093, 0.11695606, 0.1266346],
    'nrmse': [0.5052173, 0.54373233, 0.32421238, 0.35469661],
    'sliced-ws': [0.08026755, 0.093476, 0.04535191, 0.05584367]
}
for metric in ['rmse', 'nrmse', 'sliced-ws']:
    assert np.allclose(evaluator.results['imp_quality'][metric], expected_results[metric], rtol=1e-6)
print("Passed all imputation quality evaluation validation checks!")


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

expected_results = {
    'accuracy': [0.9026548672566371, 0.9026548672566371, 0.8495575221238938, 0.8230088495575221],
    'f1': [0.8705882352941177, 0.8493150684931506, 0.8089887640449438, 0.7777777777777778],
    'auc': [0.9889758179231863, 0.9615931721194879, 0.947724039829303, 0.9711948790896159],
    'prc': [0.9755904395385435, 0.9358281990553767, 0.9056074346749368, 0.9664705546556438]
}
for metric in ['accuracy', 'f1', 'auc', 'prc']:
    assert np.allclose(
        evaluator.results['local_pred'][metric],
        expected_results[metric],
        rtol=LOCAL_PREDICTION_RTOL,
        atol=LOCAL_PREDICTION_ATOL,
    )
print("Passed all local prediction evaluation validation checks!")

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
expected_results_global = {
    'accuracy': np.float64(0.908), 
    'f1': np.float64(0.8736263736263736), 
    'auc': np.float64(0.978614215793703), 
    'prc': np.float64(0.952616445725458)
}

expected_results_personalized = { 
    'accuracy': [np.float64(0.911504424778761), np.float64(0.9203539823008849), np.float64(0.8938053097345132), np.float64(0.8761061946902655)], 
    'f1': [np.float64(0.8780487804878049), np.float64(0.891566265060241), np.float64(0.8571428571428571), np.float64(0.8333333333333334)], 
    'auc': [np.float64(0.980796586059744), np.float64(0.9957325746799431), np.float64(0.9669274537695591), np.float64(0.9719061166429588)], 
    'prc': [np.float64(0.9531852351975345), np.float64(0.9915268803554507), np.float64(0.9446724031437748), np.float64(0.9599141311832291)]
}

for metric in ['accuracy', 'f1', 'auc', 'prc']:
    assert np.isclose(
        evaluator.results['fed_pred']['global'][metric],
        expected_results_global[metric],
        rtol=PREDICTION_RTOL,
        atol=PREDICTION_ATOL,
    )
    assert np.allclose(
        evaluator.results['fed_pred']['personalized'][metric],
        expected_results_personalized[metric],
        rtol=PREDICTION_RTOL,
        atol=PREDICTION_ATOL,
    )
print("Passed all federated prediction evaluation validation checks!")

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
