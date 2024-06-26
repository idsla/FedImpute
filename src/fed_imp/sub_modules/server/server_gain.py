import timeit
from copy import deepcopy
import numpy as np
import torch
from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.fed_imp.sub_modules.client.client_gain import ClientGAIN
from typing import Dict, List
from loguru import logger

from src.modules.evaluation.imputation_quality import sliced_ws
from src.tracker.EPMTracker import EMPTracker, ClientInfo, EMPRecord


class ServerGAIN:

    def __init__(
            self,
            clients: Dict[int, ClientGAIN],
            strategy_imp: StrategyImputation,
            server_config: dict,
            pred_config: dict,
            test_data: np.ndarray = None,
            seed: int = 21,
            track=False,
            run_prediction=False,
            persist_data=False
    ):

        # Basic setup
        self.clients = clients
        self.strategy_imp = strategy_imp
        self.max_workers = None
        self.config = server_config
        self.seed = seed

        # imputation parameters
        self.global_rounds_imp = server_config.get('imp_round', 100)
        self.local_rounds_imp = server_config.get('imp_local_epochs', 10)
        self.verbose = server_config['verbose']

        # group clients
        self.client_groups = {}

        ###########################################################################
        # Prediction model
        ###########################################################################
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
        avg_rmse = np.array([item['imp@rmse'] for item in rets['metrics'].values()]).mean()
        logger.info(
            "Server Round: 0 avg_rmse: {}".format(avg_rmse)
        )

        ###############################################################################################
        # N rounds imputation
        ###############################################################################################
        for current_round in range(1, self.global_rounds_imp + 1):
            if current_round % 50 == 0 or current_round == 1:
                logger.info("=" * 50)
                logger.info("Imputation Round {}".format(current_round))

            self._run_round_impute(
                server_round=current_round, client_imp_history=clients_imp_history
            )

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
        else:
            imputed_datas, origin_datas, missing_masks, test_data = None, None, None, None
            split_indices = None

        ###############################################################################################
        # Post processing
        ###############################################################################################
        imp_results = [item[2]['metrics'] for item in clients_imp_history[-5:]]

        global_ws = []
        for origin_data, impute_data in zip(origin_datas, imputed_datas):
            origin_data = origin_data[:, :-1]
            impute_data = impute_data[:, :-1]
            ws = sliced_ws(origin_data, impute_data)
            global_ws.append(ws)

        global_ws = np.array(global_ws).mean()

        return {
            'client_imp_history': clients_imp_history,
            'imp_result': {
                'imp@rmse': np.array([[value['imp@rmse'] for value in item.values()] for item in imp_results]).mean(),
                'imp@ws': np.array([[value['imp@w2'] for value in item.values()] for item in imp_results]).mean(),
                'imp@sliced_ws': np.array(
                    [[value['imp@sliced_ws'] for value in item.values()] for item in imp_results]
                ).mean(),
                'imp@global_ws': global_ws,
            },
            'pred_result': {
                'accu_mean': 0.0,
                'f1_mean': 0.0,
            },
            'data': {
                'imputed_data': imputed_datas,
                'origin_data': origin_datas,
                'missing_mask': missing_masks,
                'test_data': test_data,
                'split_indices': split_indices,
            }
        }

    def prediction(self, seeds=None):

        best_accus, best_f1s = [], []

        return best_accus, best_f1s

    ####################################################################################################################
    # Imputation
    ####################################################################################################################
    def _run_round_impute(self, server_round, client_imp_history):

        ###########################################################################################
        # One-shot local imp model training and aggregation weights
        ###########################################################################################
        # print("server round: {}".format(server_round))
        # print("====" * 20)

        ################################################################################################################
        # fit local imputation model
        ################################################################################################################
        weights, losses, missing_infos = {}, {}, {}
        if self.strategy_imp.strategy.startswith('central'):
            client = list(self.clients.values())[-1]
            client_id = list(self.clients.keys())[-1]
            weight, loss, missing_info = client.fit(
                fit_instruction={'local_epoches': self.local_rounds_imp}
            )
            weights[client_id] = weight
            losses[client_id] = loss
            missing_infos[client_id] = missing_info
        else:
            for client_id, client in self.clients.items():
                weight, loss, missing_info = client.fit(
                    fit_instruction={'local_epoches': self.local_rounds_imp}
                )
                weights[client_id] = weight
                losses[client_id] = loss
                missing_infos[client_id] = missing_info

        ################################################################################################################
        # aggregate client weights
        ################################################################################################################
        aggregated_weight, w = self.strategy_imp.aggregate(
            weights, losses, missing_infos, None, None, None, round=server_round
        )

        ################################################################################################################
        # update local imputation model
        ################################################################################################################
        if isinstance(aggregated_weight, list):
            update_weights = True
            for client_id, client in self.clients.items():
                client.transform(
                    transform_task='update_imp_model', transform_instruction={'update_weights': update_weights},
                    global_weights=aggregated_weight[client_id]
                )
        else:
            update_weights = True
            for client_id, client in self.clients.items():
                client.transform(
                    transform_task='update_imp_model', transform_instruction={'update_weights': update_weights},
                    global_weights=deepcopy(aggregated_weight)
                )

        ################################################################################################################
        # imputation
        ################################################################################################################
        if server_round <= 5 or server_round % self.verbose == 0 or server_round >= self.global_rounds_imp - 4:
            for client_id, client in self.clients.items():
                client.transform(
                    transform_task='impute_data', transform_instruction={}, global_weights=aggregated_weight
                )

            # evaluation
            rets = self._imp_evaluation(self.clients)
            client_imp_history.append(('server', server_round, rets))

            avg_rmse = np.array([item['imp@rmse'] for item in rets['metrics'].values()]).mean()
            avg_g_loss = np.array([item['g_loss'] for item in losses.values()]).mean()
            avg_d_loss = np.array([item['d_loss'] for item in losses.values()]).mean()
            logger.info(
                "Server Round: {} avg_rmse: {} avg_g_loss: {} avg_d_loss: {}".format(
                    server_round, avg_rmse, avg_g_loss, avg_d_loss)
            )

    @staticmethod
    def _imp_evaluation(clients):
        rets = {"metrics": {}, "loss": {}}
        for client_id, client in clients.items():
            ret = client.evaluate(evaluate_task='evaluate_imputation_model', evaluate_instruction={})
            rets["metrics"][client_id] = ret
        return rets