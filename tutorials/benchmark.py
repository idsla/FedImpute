
# %%
import numpy as np
import pandas as pd
import random
import os

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'

# %% [markdown]
# # Load Data and Scenario
from fedimpute.data_prep import load_data, display_data
data, data_config = load_data("codrna")
display_data(data)
print("Data Dimensions: ", data.shape)
print("Data Config:\n", data_config)

from fedimpute.scenario import ScenarioBuilder
scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_simulated_scenario(
    data, data_config, num_clients = 4, dp_strategy='iid-even', ms_scenario='mnar-heter'
)
scenario_builder.summarize_scenario()

# %% [markdown]
# # Benchmarking Pipeline
from fedimpute.pipeline import FedImputePipeline
print("Setting up pipeline...")
pipeline = FedImputePipeline()
pipeline.setup(
    id = 'benchmark_demo',
    fed_imp_configs = [
        ('em', ['local', 'fedem'], {}, [{}, {}]),
        ('mice', ['local', 'fedmice'], {}, [{}, {}]),
        ('gain', ['local', 'fedavg'], {}, [{}, {}]),
    ],
    evaluation_params = {
        'metrics': ['imp_quality', 'local_pred', 'fed_pred'],
        'model': 'lr',
    },
    persist_data = False,
    description = 'benchmark demonstration'
)

pipeline.pipeline_setup_summary()
pipeline.run_pipeline(
    scenario_builder, repeats = 5, verbose = 0
)

# %% [markdown]
# ## Result Analysis
print("Plotting results...")
import matplotlib.pyplot as plt
plt.rc('pdf', fonttype = 42)
plt.rc('ps', fonttype = 42)

pipeline.plot_pipeline_results(
    metric_aspect = 'imp_quality',    
    plot_type = 'bar',
    plot_params = {'font_size': 20, 'bar_width': 0.2},
    save_path = "./plots/benchmark_impquality.png",
    dpi = 300
)

pipeline.plot_pipeline_results(
    metric_aspect = 'local_pred',
    plot_type = 'bar',
    plot_params = {'font_size': 20, 'bar_width': 0.2},
    save_path = "./plots/benchmark_localpred.png",
    legend = False,
    dpi = 300
)

pipeline.plot_pipeline_results(
    metric_aspect = 'fed_pred_global',    
    plot_type = 'bar',
    plot_params = {'font_size': 20, 'bar_width': 0.2},
    save_path = "./plots/benchmark_fedpredglobal.png",
    legend = False,
    dpi = 300
)

