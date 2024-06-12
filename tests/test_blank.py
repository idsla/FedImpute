"""
This is a blank test file to be used as a template for new test files.
"""

"""
# Simulation Module
from fedimpute.simulator import Simulator

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
data_config = {}
scenario_params = {}

simulator = Simulator(scenario_params)

clients_local_data, global_test = simulator.simulate_scenario(data, data_config, scenario_params, seed=0)

simulator.save_scenario()
simulator.summarize_scenario()
simulator.plot_scenario()
"""

"""
# Federated Imputation Execution Environment
from fedimpute.execution_environment import FedImpEnvironment

env = FedImpEnvironment(params)

env.setup_env()
env.setup_env_from_simulator(simulator)
env.run_fed_imp()
env.save_env()
env.load_env()
"""

"""
# Evaluation
from fedimpute.evaluation import Evaluator

eval = Evaluator()
eval.evaluate_imp()
eval.evaluate_pred()
eval.evaluate_fed_pred()
"""

"""
# benchmark whole flow
from fedimpute.benchmark import Benchmark

benchmark = Benchmark(params)
benchmark.run_benchmark()
"""