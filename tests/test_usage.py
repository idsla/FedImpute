import numpy as np
import loguru
import sys

if __name__ == '__main__':
    data = np.random.rand(10000, 10)
    data_config = {
        'task_type': 'regression',
        'clf_type': None,
        'num_cols': 9,
    }

    from fedimpute.simulator import Simulator
    from fedimpute.execution_environment import FedImputeEnv

    simulator = Simulator(debug_mode=False)
    simulation_results = simulator.simulate_scenario(data, data_config, num_clients=10, verbose=1)

    env = FedImputeEnv()
    env.reset_env()
    env.configuration(imputer='miwae', fed_strategy='fedadagrad', fit_mode='fed')
    env.setup_from_simulator(simulator=simulator, verbose=1)
    print(env.imputer_name, env.fed_strategy_name)

    env.run_fed_imputation(run_type='parallel')
