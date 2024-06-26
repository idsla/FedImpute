import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
import argparse
from itertools import product
import yaml
from copy import deepcopy
import random
import multiprocessing as mp
import json

from src.modules.data_spliting import split_train_test
from src.modules.data_preprocessing import load_data
from src.fed_imp.experiment import main_func

results = []
# PARAMS
ALPHAS = [0.5, 0.65, 0.8, 0.95]
SFS = [2, 4]
GAMMAS = [0.05, 0.3]
Ns = [3, 5, 7, 9, 11]
Rs = [0.0, 0.5, 0.9, 1.0]
import sys
import traceback


def main(configuration):
    # general parameters
    num_clients = configuration['num_clients']

    # set random seed
    seed = configuration['experiment']['seed']
    mtp = False
    tune_params = configuration['tune_params']
    random.seed(seed)  # seed for split data

    # load data
    dataset_params = configuration['data']
    data, data_config = load_data(**dataset_params)
    regression = data_config['task_type'] == 'regression'

    n_rounds = configuration['experiment']['n_rounds']
    test_size = configuration['experiment'].get('test_size', 0.1)
    n_rounds = 1
    if n_rounds == 1:
        n_rounds_data = split_train_test(data, n_folds=2, seed=seed, test_size=test_size, regression=regression)
    else:
        n_rounds_data = split_train_test(
            data, n_folds=n_rounds, seed=seed, test_size=test_size, regression=regression
        )

    # n rounds average
    train_data, test_data = n_rounds_data[0]

    results = []
    if mtp:
        seed = configuration['experiment']['random_seed']
        seeds = [(seed + 10087 * i) % (2 ^ 23) for i in range(n_rounds)]
        rounds = list(range(n_rounds))
        # multiprocessing
        num_processes = configuration['experiment']['num_process']
        chunk_size = n_rounds // num_processes
        if chunk_size == 0:
            chunk_size = 1
        # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

        # fed_imp start
        with mp.Pool(num_processes) as pool:
            process_args = [
                (train_data, test_data, configuration, num_clients, data_config, round, seed)
                for round, seed in zip(rounds, seeds)]
            process_results = pool.starmap(main_func, process_args, chunksize=chunk_size)

        for ret in process_results:
            results.extend(ret[0])
    else:
        seed = configuration['experiment']['random_seed']
        seeds = [(seed + 10087 * i) % (2 ^ 23) for i in range(n_rounds)]
        rounds = list(range(n_rounds))
        for round, seed in zip(rounds, seeds):
            rets, stat_tracker = main_func(
                train_data, test_data, configuration, num_clients, data_config, round, seed
            )

            for ret in rets:
                results.append(ret)

    return results[0]['imp_result']['imp@rmse'], results[0]['imp_result']['imp@sliced_ws']


if __name__ == '__main__':

    ####################################################################################################################
    # command argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='name of dataset')
    parser.add_argument('--s', type=str, default=None, help='scenario')
    parser.add_argument('--mm', type=str, default=None, help='mm type')
    parser.add_argument('--tmpl', type=str, default='imp_pc2', help='name of tmpl')
    parser.add_argument('--np', type=int, default=16, help='number of processes')
    args = parser.parse_args()

    ####################################################################################################################
    # args
    scenario = args.s
    if scenario not in ['s1', 's2']:  # s1, s2
        raise ValueError('scenario not supported')
    mm = args.mm
    if mm not in ['lr', 'rl', 'even']:  # lr, rl, even
        raise ValueError('mm type not supported')
    tmpl_name = args.tmpl
    dataset = args.dataset
    p_size = 1000 if dataset != 'codon' else 600

    ####################################################################################################################
    # read template file
    experiment_config_template_path = f'conf/config_tmpl/{tmpl_name}.yaml'
    with open(experiment_config_template_path, 'r') as f:
        experiment_config_template = yaml.safe_load(f)

    ####################################################################################################################
    # loads all experiments
    s_params = None
    if scenario == 's1':
        s_params = Ns
    else:
        s_params = Rs

    param_grid = list(product(s_params, GAMMAS, ALPHAS, SFS))  # [(N or r, alpha, sf), ...]
    configs = []
    for param in param_grid:
        if scenario == 's1':
            N, gamma, alpha, sf = param
            r = None
        else:
            r, gamma, alpha, sf = param
            N = 10

        config = deepcopy(experiment_config_template)
        config['data']['dataset_name'] = args.dataset
        if scenario == 's1':
            config['num_clients'] = N
        config['missing_simulate']['mr_strategy'] = 'fixed@mr=0.5'
        config['missing_simulate']['mf_strategy'] = 'all'
        if scenario == 's1':
            if mm == 'lr':
                config['missing_simulate']['mm_strategy'] = 'mnar_lr@sp=extremel1'
                config['data_partition']['strategy'] = f'sample-unevenl1-{p_size}'
            else:
                config['missing_simulate']['mm_strategy'] = 'mnar_rl@sp=extremer1'
                config['data_partition']['strategy'] = f'sample-unevenr1-{p_size}'
        else:  # 's2-0.5'
            if mm == 'even':
                config['missing_simulate']['mm_strategy'] = 'mnar_lr@sp=extreme_r={}'.format(r)
                config['data_partition']['strategy'] = f'sample-evenly'
            else:
                raise ValueError('mm type not supported')

        config['agg_strategy_imp']['strategy'] = 'fedmechw_new'
        config["algo_params"]["fedmechw_new"] = {
            "alpha": alpha,
            "gamma": gamma,
            "client_thres": 1.0,
            "scale_factor": sf
        }

        configs.append(config)

    print(f'Num of configs: {len(configs)}')
    ####################################################################################################################
    # run experiments parallel
    num_processes = max(1, mp.cpu_count())  # Safe fallback to prevent division by zero

    # multiprocessing
    with mp.Pool(num_processes) as pool:
        process_results = pool.map(main, configs)  # Let Pool handle the chunking

    print(process_results)

    param_result = {}

    for param, result in zip(param_grid, process_results):
        param_result[param] = result
    dir = f'./results/raw_results/hyper_params_tune/'
    import os
    import pickle

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + f'{dataset}_{scenario}_{mm}_{tmpl_name}.pkl', 'wb') as f:
        pickle.dump(param_result, f)
