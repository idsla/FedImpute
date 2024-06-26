import timeit
from copy import deepcopy
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.fed_imp.sub_modules.client.client import Client
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
from typing import Dict, List
from loguru import logger
from src.tracker.EPMTracker import EMPTracker, ClientInfo, EMPRecord
from copy import deepcopy


def client_fit(args):
    client, client_id, col_idx = args
    weight, loss, missing_info, projection_matrix, ms_coef = client.fit(
        fit_task='fit_imputation_model', fit_instruction={'feature_idx': col_idx}
    )
    return client_id, weight, loss, missing_info, projection_matrix, ms_coef, client


def process_args(args):
    return client_fit(*args)


class ServerBase:

    def __init__(
            self,
            clients: Dict[int, Client],
            strategy_imp: StrategyImputation,
            server_config: dict,
            pred_config: dict,
            test_data: np.ndarray = None,
            seed: int = 21,
            track=False,
            run_prediction=True,
            persist_data=False
    ):

        # Basic setup
        self.clients = clients
        self.strategy_imp = strategy_imp
        self.max_workers = None
        self.config = server_config
        self.seed = seed

        # imputation parameters
        self.num_rounds_imp = server_config.get('imp_round', 30)
        self.imp_model_fit_mode = server_config.get('model_fit_mode', "one_shot")
        self.froze_ms_coefs_round = server_config.get('froze_ms_coefs_round', 1)

        # group clients
        self.client_groups = {}

        ###########################################################################
        # Prediction model
        ###########################################################################
        self.run_prediction = run_prediction
        # test data
        self.test_data = test_data

        ###########################################################################
        # Experiment Tracker
        ###########################################################################
        self.persist_data = persist_data
        self.stats_tracker = EMPTracker()
        self.track = track
        if track:
            for client in self.clients.values():
                self.stats_tracker.client_infos.append(
                    ClientInfo(
                        client_id=client.client_id, missing_mask=client.missing_mask,
                        data_true=np.concatenate([client.X_train, client.y_train.reshape(-1, 1)], axis=1)
                    )
                )

        self.ms_coefs = dict()

    def run(self):

        # Start Federated Learning
        torch.manual_seed(self.seed)

        start_time = timeit.default_timer()
        clients_imp_history, clients_prediction_history1, clients_prediction_history2 = [], [], []

        ###############################################################################################
        # Collection information's from clients
        ###############################################################################################
        # get initial information from clients
        initial_values_mean, sample_sizes, missing_ratios = [], [], []
        initial_values_max, initial_values_min = [], []
        for client_id, client in self.clients.items():
            mean_values, max_values, min_values, sample_size, ms_ratio = client.get_initial_values()
            initial_values_mean.append(mean_values)
            initial_values_max.append(max_values)
            initial_values_min.append(min_values)
            missing_ratios.append(ms_ratio)
            sample_sizes.append(sample_size)

        ###############################################################################################
        # Group clients based on missing ratio
        ###############################################################################################
        self.client_groups = self.clustering(sample_sizes)

        ###############################################################################################
        # initial imputation and evaluation
        ###############################################################################################
        # global min and max
        if self.strategy_imp.strategy != 'local':
            initial_values_max = np.array(initial_values_max)
            initial_values_min = np.array(initial_values_min)
            global_max = initial_values_max.max(axis=0, initial=1)
            global_min = initial_values_min.min(axis=0, initial=0)
            for client in self.clients.values():
                client.features_max = global_max
                client.features_min = global_min

        for client in self.clients.values():
            client.update_imp_model_minmax()

        # initial global imputation
        aggregated_initial_values_mean = self.strategy_imp.aggregate_initial(
            initial_values_mean, sample_sizes, missing_ratios, self.client_groups
        )

        for client_id, client in self.clients.items():
            if isinstance(aggregated_initial_values_mean, dict):
                client.initial_impute(aggregated_initial_values_mean[client_id])
            else:
                client.initial_impute(aggregated_initial_values_mean)

            # stats tracking
            if self.track:
                self.stats_tracker.client_infos[client_id].data_imp.append(
                    np.concatenate([client.X_train_filled, client.y_train.reshape(-1, 1)], axis=1)
                )

        rets = self._imp_evaluation(self.clients)
        clients_imp_history.append(('server', 0, rets))

        ###############################################################################################
        # N rounds imputation
        ###############################################################################################
        final_local_coefs, final_mm_ceofs = None, None
        for current_round in range(1, self.num_rounds_imp + 1):
            if current_round % 10 == 0 or current_round == 1:
                logger.info("=" * 50)
                logger.info("Imputation Round {}".format(current_round))

            imp_ret_dict = self._run_round_impute(
                server_round=current_round, client_imp_history=clients_imp_history, total_rounds=self.num_rounds_imp
            )

            if current_round == self.num_rounds_imp:
                final_local_coefs, final_mm_ceofs = imp_ret_dict['local_coefs'], imp_ret_dict['mm_coefs']

        ###################################################################################################
        # Prediction
        ###################################################################################################
        if self.run_prediction:
            ###############################################################################################
            # Prediction FedAvg
            ###############################################################################################
            best_accus, best_f1s = self.prediction()

            logger.info(
                "model1 test acc: {:.6f} ({:.3f}), test f1: {:.6f} ({:.3f})".format(
                    np.array(best_accus).mean(), np.array(best_accus).std(), np.array(best_f1s).mean(),
                    np.array(best_f1s).std()
                )
            )
        else:
            best_accus, best_f1s = [], []

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        logger.info("FL finished in {}".format(elapsed))

        ###############################################################################################
        # save original data and imputed data

        if self.persist_data:
            test_data, imputed_datas, origin_datas, missing_masks = self.test_data, [], [], []
            for client in self.clients.values():
                imputed_data = np.concatenate([client.X_train_filled, client.y_train.reshape(-1, 1)], axis=1)
                origin_data = np.concatenate([client.X_train, client.y_train.reshape(-1, 1)], axis=1)
                missing_mask = client.missing_mask
                imputed_datas.append(imputed_data)
                origin_datas.append(origin_data)
                missing_masks.append(missing_mask)
            split_indices = np.cumsum([item.shape[0] for item in imputed_datas])[:-1]
            imputed_datas = np.concatenate(imputed_datas, axis=0)
            origin_datas = np.concatenate(origin_datas, axis=0)
            missing_masks = np.concatenate(missing_masks, axis=0)
            local_coefs = final_local_coefs
            mm_coefs = final_mm_ceofs
        else:
            imputed_datas, origin_datas, missing_masks, test_data = None, None, None, None
            local_coefs, mm_coefs = None, None

        ###############################################################################################
        # Post processing
        ###############################################################################################
        imp_results = [item[2]['metrics'] for item in clients_imp_history[-5:]]

        return {
            'client_imp_history': clients_imp_history,
            'imp_result': {
                'imp@rmse': np.array([[value['imp@rmse'] for value in item.values()] for item in imp_results]).mean(),
                'imp@ws': np.array([[value['imp@w2'] for value in item.values()] for item in imp_results]).mean(),
                'imp@sliced_ws': np.array(
                    [[value['imp@sliced_ws'] for value in item.values()] for item in imp_results]
                ).mean(),
            },
            'pred_result': {
                'accu_mean': np.array(best_accus).mean() if len(best_accus) > 0 else 0.0,
                'f1_mean': np.array(best_f1s).mean() if len(best_f1s) > 0 else 0.0,
            },
            'data': {
                'imputed_data': imputed_datas,
                'origin_data': origin_datas,
                'missing_mask': missing_masks,
                'test_data': test_data,
                'split_indices': split_indices,
                'local_coef': local_coefs,
                'mm_coef': mm_coefs,
            }
        }

    def prediction(self, seeds=None):

        best_accus, best_f1s = [], []

        return best_accus, best_f1s

    ####################################################################################################################
    # Imputation
    ####################################################################################################################
    def _run_round_impute(self, server_round, client_imp_history, total_rounds=30):

        ###########################################################################################
        # One-shot local imp model training and aggregation weights
        ###########################################################################################
        # print("server round: {}".format(server_round))
        # print("====" * 20)
        imp_info_dict = self._imp_round_instant(
            num_cols=self.config['n_cols'], clients=self.clients,
            strategy_imp=self.strategy_imp, aggregate=True, server_round=server_round
        )

        # check convergence
        for client_id, client in self.clients.items():
            client.check_convergency()
            # stats tracking
            if self.track:
                self.stats_tracker.client_infos[client_id].data_imp.append(
                    np.concatenate([client.X_train_filled, client.y_train.reshape(-1, 1)], axis=1)
                )

        # evaluation
        if server_round % 1 == 0 or server_round >= (total_rounds - 3):
            rets = self._imp_evaluation(self.clients)
            client_imp_history.append(('server', server_round, rets))

        return imp_info_dict

    ####################################################################################################################
    # Imputation Utils function
    ####################################################################################################################
    def _imp_round_instant(self, num_cols, clients, strategy_imp, aggregate=True, server_round=None):

        local_coefs, mm_coefs = [], []
        for col_idx in range(num_cols):

            if server_round <= self.froze_ms_coefs_round:
                weights, losses, missing_infos, proj_matrix, ms_coefs, top_k_idx_clients = {}, {}, {}, {}, {}, {}
                for client_id, client in clients.items():
                    weight, loss, missing_info, projection_matrix, ms_coef = client.fit(
                        fit_task='fit_imputation_model', fit_instruction={'feature_idx': col_idx}
                    )
                    weights[client_id] = weight
                    losses[client_id] = loss
                    missing_infos[client_id] = missing_info
                    proj_matrix[client_id] = projection_matrix
                    ms_coefs[client_id] = ms_coef
                    top_k_idx_clients[client_id] = client.top_k_idx

                # persist ms_coefs
                self.ms_coefs[col_idx] = ms_coefs
            else:
                weights, losses, missing_infos, proj_matrix, ms_coefs, top_k_idx_clients = {}, {}, {}, {}, {}, {}
                for client_id, client in clients.items():
                    weight, loss, missing_info, projection_matrix, ms_coef = client.fit(
                        fit_task='fit_imputation_model', fit_instruction={'feature_idx': col_idx}
                    )
                    weights[client_id] = weight
                    losses[client_id] = loss
                    missing_infos[client_id] = missing_info
                    proj_matrix[client_id] = projection_matrix
                    top_k_idx_clients[client_id] = client.top_k_idx

                # fetch stored ms_coefs
                ms_coefs = self.ms_coefs[col_idx]

            # aggregate client weights
            if aggregate:
                aggregated_weight, w = strategy_imp.aggregate(
                    weights, losses, missing_infos, self.client_groups[col_idx], proj_matrix, ms_coefs,
                    round=server_round
                )
            else:
                aggregated_weight, w = None, None

            # distributed aggregated weights
            if isinstance(aggregated_weight, list):
                for client_id, client in clients.items():
                    client.transform(
                        transform_task='impute_data', transform_instruction={'feature_idx': col_idx},
                        global_weights=aggregated_weight[client_id], update_weights=aggregate
                    )
            else:
                for client_id, client in clients.items():
                    client.transform(
                        transform_task='impute_data', transform_instruction={'feature_idx': col_idx},
                        global_weights=aggregated_weight, update_weights=aggregate
                    )

            # stats tracking
            weights_new = [weights[client_id] for client_id in range(0, len(clients))]
            losses_new = [losses[client_id] for client_id in range(0, len(clients))]
            ms_coefs_new = [ms_coefs[client_id] for client_id in range(0, len(clients))]
            if self.track:
                self.stats_tracker.records.append(
                    EMPRecord(
                        iteration=server_round, feature_idx=col_idx,
                        aggregation_weights=np.array(w).copy(),
                        global_model_params=np.array(aggregated_weight).copy(),
                        local_imp_model_params=np.array(weights_new).copy(),
                        local_mm_model_params=np.array(ms_coefs_new).copy(),
                        losses=np.array(losses_new).copy(),
                    )
                )

            local_coefs.append(weights_new)
            mm_coefs.append(ms_coefs_new)

        return {
            'local_coefs': np.stack(local_coefs),
            "mm_coefs": np.stack(mm_coefs)
        }

    @staticmethod
    def _imp_evaluation(clients):
        rets = {"metrics": {}, "loss": {}}
        for client_id, client in clients.items():
            ret = client.evaluate(evaluate_task='evaluate_imputation_model', evaluate_instruction={})
            rets["metrics"][client_id] = ret
        return rets

    @staticmethod
    def clustering(sample_sizes: List[np.ndarray]) -> Dict[int, List[List[int]]]:
        # sample_size -> (num_clients, num_features)
        client_groups = {}
        sample_sizes = np.array(sample_sizes).T
        for feature_idx in range(sample_sizes.shape[0]):
            sample_size = sample_sizes[feature_idx]
            if len(sample_size) == 1:
                cluster_labels = np.zeros(len(sample_size), dtype=np.int32)
            else:
                agg = AgglomerativeClustering(
                    n_clusters=None, metric='l1', linkage='average', distance_threshold=0.05
                )
                cluster_labels = agg.fit_predict(sample_size.reshape(-1, 1))
            groups = [[] for _ in range(len(set(cluster_labels)))]
            for i in range(len(sample_size)):
                cluster_idx = cluster_labels[i]
                groups[cluster_idx].append(i)
            client_groups[feature_idx] = deepcopy(groups)

        return client_groups
