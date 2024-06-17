from typing import List, Union
import numpy as np
from fedimpute.execution_environment.client import Client
from fedimpute.evaluation.imp_quality_metrics import rmse, sliced_ws


class Evaluator:

    def __init__(self, evaluator_params: dict = None):
        if evaluator_params is None:
            evaluator_params = {}
        self.evaluator_params = evaluator_params

    @staticmethod
    def evaluate_imputation_local(
            X_train_imp: np.ndarray, X_train_origin: np.ndarray, X_train_mask: np.ndarray
    ) -> dict:

        imp_rmse = rmse(X_train_imp, X_train_origin, X_train_mask)
        imp_ws = sliced_ws(X_train_imp, X_train_origin)
        return {
            'imp_rmse': imp_rmse,
            'imp_ws': imp_ws
        }

    @staticmethod
    def evaluate_imputation(
            X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], X_train_masks: List[np.ndarray],
            central_client: bool = False
    ) -> dict:

        evaluation_results = {
            'imp_rmse_clients': [],
            'imp_ws_clients': [],
            'imp_rmse_avg': 0,
            'imp_ws_avg': 0,
            'imp_rmse_global': 0,
            'imp_ws_global': 0
        }

        # imputation quality evaluation
        for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
            imp_rmse = rmse(X_train_imp, X_train_origin, X_train_mask)
            imp_ws = sliced_ws(X_train_imp, X_train_origin)
            evaluation_results['imp_rmse_clients'].append(imp_rmse)
            evaluation_results['imp_ws_clients'].append(imp_ws)

        if central_client:
            evaluation_results['imp_rmse_avg'] = float(np.mean(evaluation_results['imp_rmse_clients'][:-1]))
            evaluation_results['imp_ws_avg'] = float(np.mean(evaluation_results['imp_ws_clients'][:-1]))
        else:
            evaluation_results['imp_rmse_avg'] = float(np.mean(evaluation_results['imp_rmse_clients']))
            evaluation_results['imp_ws_avg'] = float(np.mean(evaluation_results['imp_ws_clients']))

        # global imputation quality evaluation
        # merged_X_imp = np.concatenate(X_train_imps, axis=0)
        # merged_X_origin = np.concatenate(X_train_origins, axis=0)
        # merged_X_mask = np.concatenate(X_train_masks, axis=0)
        # imp_rmse = rmse(merged_X_imp, merged_X_origin, merged_X_mask)
        # imp_ws = sliced_ws(merged_X_imp, merged_X_origin)
        # evaluation_results['imp_rmse_global'] = imp_rmse
        # evaluation_results['imp_ws_global'] = imp_ws

        return evaluation_results

    @staticmethod
    def get_imp_quality(evaluation_results: dict, individual: bool = True, metric='rmse') -> Union[dict, float]:

        if individual is True:
            return evaluation_results[f'imp_{metric}_clients']
        else:
            return evaluation_results[f'imp_{metric}_avg']

    @staticmethod
    def evaluation_prediction(clients: List[Client]):
        pass
