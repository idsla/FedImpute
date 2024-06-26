from copy import deepcopy

import numpy as np
from src.modules.evaluation.imputation_quality import sliced_ws
from src.fed_imp.sub_modules.server.load_server import load_server
from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.fed_imp.sub_modules.client.client_factory import ClientsFactory
import random
from src.modules.data_partition import data_partition
from src.modules.data_spliting import split_train_test
from src.modules.data_preprocessing import load_data
from src.fed_imp.sub_modules.missing_simulate.missing_adder_new import add_missing
from src.fed_imp.sub_modules.result_processing import (
    processing_imputation_result, visualizing_clients_result,
    average_n_rounds_result,
)
from loguru import logger
import multiprocessing as mp
import itertools
from config import settings
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from src.hyper_params import Hyperparameters


def main_func(
        train_data, test_data, configuration, num_clients, data_config, round, seed,
        param=None
):
    logger.info(f"Round {round}")
    if param is None:
        repeats = 1
    else:
        repeats = 2

    rets, stat_trackers = [], []
    for repeat in range(repeats):
        new_seed = (seed + 10087 * repeat)
        #####################################################################################################
        # Create clients
        #####################################################################################################
        # data partition
        regression = data_config['task_type'] == 'regression'
        data_partition_params = configuration['data_partition']
        data_partitions = data_partition(
            **data_partition_params, data=train_data.values, n_clients=num_clients, seed=new_seed,
            regression=regression
        )

        #####################################################################################################
        # missing data simulate
        missing_params = configuration['missing_simulate']
        cols = np.arange(0, train_data.shape[1] - 1)
        scenario = missing_params
        data_ms_clients = add_missing(
            train_data_list=data_partitions, scenario=scenario, cols=cols, seed=new_seed
        )

        client_factory = ClientsFactory(debug=False)
        client_type = configuration['client_type']
        clients = client_factory.generate_clients(
            num_clients, data_partitions, data_ms_clients, test_data.values, data_config,
            configuration['imputation'], client_type=client_type, seed=new_seed
        )

        #####################################################################################################
        # Create Strategy
        #####################################################################################################
        # Create Imputation Strategy
        imp_strategy = configuration['agg_strategy_imp']['strategy']
        strategy_name = imp_strategy.split('-')[0]
        if param:
            params = param
            print(f"Used setted params: {params}")
        else:
            hyper_params = Hyperparameters(
                dataset=configuration['data']['dataset_name'],
                data_partition=configuration['data_partition']['strategy'],
                mm_strategy=configuration['missing_simulate']['mm_strategy'],
                num_clients=configuration['num_clients'],
                method=strategy_name
            )
            default_params = hyper_params.get_params()  # get tuned params
            if default_params is None:  # if no tuned params, use default params
                try:
                    params = configuration['algo_params'][strategy_name]
                    print(f"Used default params: {params}")
                except:
                    raise ValueError("No params")
            else:
                params = default_params
                print(f"Used tuned params: {params}")

        strategy_imp = StrategyImputation(strategy=imp_strategy, params=params)

        # if imp_strategy == 'central':
        #     data_ms_new = [np.concatenate(data_ms_clients, axis=0)]
        #     data_partitions_new = [np.concatenate(data_partitions, axis=0)]
        #     clients = client_factory.generate_clients(
        #         1, data_partitions_new, data_ms_new, test_data.values, data_config,
        #         configuration['imputation'], seed=new_seed
        #     )

        #     assert len(clients.keys()) == 1
        #     strategy_imp = StrategyImputation(strategy='local', params={})

        if imp_strategy == 'central2':
            data_ms_new = np.concatenate(data_ms_clients, axis=0)
            data_partitions_new = np.concatenate(data_partitions, axis=0)
            data_ms_clients.append(data_ms_new)
            data_partitions.append(data_partitions_new)
            clients = client_factory.generate_clients(
                num_clients + 1, data_partitions, data_ms_clients, test_data.values, data_config,
                configuration['imputation'], seed=new_seed, client_type='ice'
            )

            assert len(clients.keys()) == num_clients + 1
        elif imp_strategy == 'central_vae':
            data_ms_new = np.concatenate(data_ms_clients, axis=0)
            data_partitions_new = np.concatenate(data_partitions, axis=0)
            data_ms_clients.append(data_ms_new)
            data_partitions.append(data_partitions_new)
            clients = client_factory.generate_clients(
                num_clients + 1, data_partitions, data_ms_clients, test_data.values, data_config,
                configuration['imputation'], seed=new_seed, client_type = 'vae'
            )

            assert len(clients.keys()) == num_clients + 1
        elif imp_strategy == 'central_gain':
            data_ms_new = np.concatenate(data_ms_clients, axis=0)
            data_partitions_new = np.concatenate(data_partitions, axis=0)
            data_ms_clients.append(data_ms_new)
            data_partitions.append(data_partitions_new)
            clients = client_factory.generate_clients(
                num_clients + 1, data_partitions, data_ms_clients, test_data.values, data_config,
                configuration['imputation'], seed=new_seed, client_type='gain')

        #####################################################################################################
        # Create Server
        server_type = configuration['server_type']
        server_config = configuration['server']
        server_config["n_cols"] = test_data.shape[1] - 1

        pred_config = configuration['pred_model']
        pred_config['model_params']['input_feature_dim'] = test_data.shape[1] - 1
        pred_config['model_params']['output_classes_dim'] = len(np.unique(test_data.iloc[:, -1].values))

        server = load_server(
            server_type,
            clients=clients,
            strategy_imp=strategy_imp,
            server_config=server_config,
            pred_config=pred_config,
            test_data=test_data.values,
            seed=new_seed,
            track=configuration['track'],
            run_prediction=configuration['prediction'],
            persist_data=configuration['save_state'],
        )

        # return server
        ret = server.run()
        rets.append(ret)

        if configuration['track']:
            stat_trackers.append(deepcopy(server.stats_tracker))

        del clients
        del server
        del strategy_imp

    return rets, stat_trackers


class Experiment:
    name = 'fed_imp'

    def __init__(self, debug: bool = False):
        self.debug = debug

    def run_experiment(self, configuration: dict):

        # general parameters
        num_clients = configuration['num_clients']

        # set random seed
        seed = configuration['experiment']['seed']
        mtp = configuration['experiment']['mtp']
        tune_params = configuration['tune_params']
        random.seed(seed)  # seed for split data

        # load data
        dataset_params = configuration['data']
        data, data_config = load_data(**dataset_params)
        regression = data_config['task_type'] == 'regression'

        n_rounds = configuration['experiment']['n_rounds']
        test_size = configuration['experiment'].get('test_size', 0.1)
        if n_rounds == 1:
            n_rounds_data = split_train_test(data, n_folds=2, seed=seed, test_size=test_size, regression=regression)
        else:
            n_rounds_data = split_train_test(
                data, n_folds=n_rounds, seed=seed, test_size=test_size, regression=regression
            )

        # n rounds average
        train_data, test_data = n_rounds_data[0]

        imbalance_strategy = configuration.get('handle_imbalance', None)
        if imbalance_strategy == 'oversampling':
            columns = train_data.columns
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            ros = RandomOverSampler(random_state=seed)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)
        elif imbalance_strategy == 'undersampling':
            columns = train_data.columns
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            ros = RandomUnderSampler(random_state=seed)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)
        elif imbalance_strategy == 'smote':
            columns = train_data.columns
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            smote = SMOTE(random_state=seed, n_jobs=-1)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)
        elif imbalance_strategy == 'adasyn':
            columns = train_data.columns
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values
            ada = ADASYN(random_state=seed, n_jobs=-1)
            X_train, y_train = ada.fit_resample(X_train, y_train)
            train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)

        print(train_data.shape)
        stat_trackers = []
        if tune_params:
            param_grid = settings['algo_params_grids'][configuration['agg_strategy_imp']['strategy']]
            print(param_grid)
            keys = param_grid.keys()
            combinations = list(itertools.product(*param_grid.values()))
            params = []
            for comb in combinations:
                params.append(dict(zip(keys, comb)))

            results = []
            # multiprocessing
            num_processes = 5
            chunk_size = len(params) // num_processes
            if chunk_size == 0:
                chunk_size = 1
            # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

            # fed_imp start
            seeds = [configuration['experiment']['random_seed'] for i in range(len(params))]
            with mp.Pool(num_processes) as pool:
                process_args = [
                    (train_data, test_data, configuration, num_clients, data_config, i, seed, param)
                    for i, param, seed in zip(range(len(params)), params, seeds)
                ]
                process_results = pool.starmap(main_func, process_args, chunksize=chunk_size)

            for ret in process_results:
                results.extend(ret[0])

        elif mtp:
            seed = configuration['experiment']['random_seed']
            seeds = [(seed + 10087 * i) for i in range(n_rounds)]
            #seeds = [i for i in range(n_rounds)]
            rounds = list(range(n_rounds))
            results = []
            # multiprocessing
            num_processes = configuration['experiment']['num_process']
            chunk_size = n_rounds // num_processes
            if chunk_size == 0:
                chunk_size = 1
            # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

            # fed_imp start
            with mp.Pool(num_processes) as pool:
                try:
                    process_args = [
                        (train_data, test_data, configuration, num_clients, data_config, round, seed)
                        for round, seed in zip(rounds, seeds)]
                    process_results = pool.starmap(main_func, process_args, chunksize=chunk_size)
                except KeyboardInterrupt:
                    print("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                else:
                    print("Normal termination")
                    pool.close()
                
            for ret in process_results:
                results.extend(ret[0])
        else:
            seed = configuration['experiment']['random_seed']
            seeds = [(seed + 10087 * i) % (2 ^ 23) for i in range(n_rounds)]
            rounds = list(range(n_rounds))
            results = []
            for round, seed in zip(rounds, seeds):
                rets, stat_tracker = main_func(
                    train_data, test_data, configuration, num_clients, data_config, round, seed
                )
                for ret in rets:
                    results.append(ret)

                stat_trackers.append(stat_tracker)

        # post analysis
        logger.info("Summary: {}".format(configuration['agg_strategy_imp']['strategy']))
        ret = self.experiment_result_processing(results, tune_params=tune_params)

        return ret, stat_trackers

    @staticmethod
    def experiment_pre_setup(configuration):
        pass

    @staticmethod
    def experiment_result_processing(results, tune_params=False):

        #####################################################################################################
        # process client imp history
        #####################################################################################################
        if tune_params:
            for result in results:
                logger.info(
                    "rmse {:3f} ws {:.3f} slicedws {:.3f} accu {:.3f} f1 {:3f}".format(
                        result['imp_result']['imp@rmse'], result['imp_result']['imp@ws'],
                        result['imp_result']['imp@sliced_ws'],
                        result['pred_result']['accu_mean'], result['pred_result']['f1_mean']
                    )
                )
            return None

        #####################################################################################################
        # imp history
        #####################################################################################################
        client_imp_rets = []
        x = []
        for result in results:
            clients_imp_ret, x = processing_imputation_result(result['client_imp_history'])
            client_imp_rets.append(clients_imp_ret)

        # average
        clients_imp_ret_avg = average_n_rounds_result(client_imp_rets)

        metrics = ['imp@rmse', 'imp@w2', "imp@sliced_ws"]
        encoded_image = visualizing_clients_result(clients_imp_ret_avg, metrics, x)

        final_rets = {}
        for metric in metrics:
            if len(clients_imp_ret_avg[metric]['client_avg']) < 5:
                last_several_rounds = 0
            else:
                last_several_rounds = -5
            final_rets[metric] = sum(clients_imp_ret_avg[metric]['client_avg'][last_several_rounds:]) / 5
            logger.info(f"{metric}: avg {final_rets[metric]}")

        #####################################################################################################
        # process client pred history
        #####################################################################################################
        #  accuracy
        imp_rmse, imp_ws, imp_sliced_ws, best_accu1, best_f11 = [], [], [], [], []
        for result in results:
            best_accu1.append(result['pred_result']['accu_mean'])
            imp_rmse.append(result['imp_result']['imp@rmse'])
            imp_ws.append(result['imp_result']['imp@ws'])
            imp_sliced_ws.append(result['imp_result']['imp@sliced_ws'])
            best_f11.append(result['pred_result']['f1_mean'])

        if np.array(best_accu1).size > 0:
            logger.info(
                "model pred_accu: {:.5f} ({:.3f}) pred_f1: {:.5f} ({:.3f})".format(
                    np.array(best_accu1).mean(), np.array(best_accu1).std(), np.array(best_f11).mean(),
                    np.array(best_f11).std()
                )
            )

        #####################################################################################################
        # DATA
        #####################################################################################################
        data_results = {}
        for result in results:
            for key in result['data'].keys():
                if key not in data_results.keys():
                    if result['data'][key] is not None:
                        data_results[key] = [result['data'][key]]
                else:
                    if result['data'][key] is not None:
                        data_results[key].append(result['data'][key])

        global_ws = compute_global_ws(data_results)

        logger.info(
            "imp@rmse: {:.5f} ({:.3f}) imp@sliced_ws: {:.5f} ({:.3f}) imp@global_ws: {:.5f}".format(
                np.array(imp_rmse).mean(), np.array(imp_rmse).std(),
                np.array(imp_sliced_ws).mean(), np.array(imp_sliced_ws).std(), global_ws
            )
        )

        return {
            'avg_rets_final': final_rets,
            'avg_imp_final': {
                'imp@rmse': np.array(imp_rmse).mean(),
                'imp@w2': np.array(imp_ws).mean(),
                'imp@sliced_ws': np.array(imp_sliced_ws).mean(),
                "imp@global_ws": global_ws,
                'imp@rmse_std': np.array(imp_rmse).std(),
                'imp@w2_std': np.array(imp_ws).std(),
                'imp@sliced_ws_std': np.array(imp_sliced_ws).std()
            },
            'avg_pred_final_model': {
                'accu': np.array(best_accu1).mean(),
                'f1': np.array(best_f11).mean(),
                'accu_std': np.array(best_accu1).std(),
                'f1_std': np.array(best_f11).std()
            },
            'clients_imp_ret_clean': clients_imp_ret_avg,
            'clients_imp_history_raw': [result['client_imp_history'] for result in results],
            'pred_results': [result['pred_result'] for result in results],
            'data': data_results,
            "plots": [encoded_image]
        }


def compute_global_ws(data_result):

    origin_datas = data_result['origin_data']
    imputed_datas = data_result['imputed_data']
    global_ws = []
    for origin_data, impute_data in zip(origin_datas, imputed_datas):
        origin_data = origin_data[:, :-1]
        impute_data = impute_data[:, :-1]
        ws = sliced_ws(origin_data, impute_data)
        global_ws.append(ws)

    return np.array(global_ws).mean()

