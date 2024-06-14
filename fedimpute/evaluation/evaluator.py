import json
from collections import OrderedDict
from copy import deepcopy
from typing import List, Dict, Union
import warnings

import loguru
import numpy as np

from .imp_quality_metrics import rmse, sliced_ws
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .twonn import TwoNNRegressor, TwoNNClassifier
from .pred_model_metrics import task_eval
from ..utils.reproduce_utils import set_seed
from ..utils.nn_utils import EarlyStopping
from fedimpute.execution_environment import FedImputeEnv

warnings.filterwarnings("ignore")
from tqdm.auto import trange


class Evaluator:

    def __init__(self):
        self.results = None

    def evaluate(self, env: FedImputeEnv, metrics: Union[List, None] = None, seed: int = 0):

        if metrics is None:
            metrics = ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed']

        for metric in metrics:
            if metric not in ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed']:
                raise ValueError(f"Invalid metric: {metric}")

        results = {}
        X_train_imps = [client.X_train_imp for client in env.clients]
        X_train_origins = [client.X_train for client in env.clients]
        X_train_masks = [client.X_train_mask for client in env.clients]
        y_trains = [client.y_train for client in env.clients]
        X_tests = [client.X_test for client in env.clients]
        y_tests = [client.y_test for client in env.clients]

        if 'imp_quality' in metrics:
            print("Evaluating imputation quality...")
            results['imp_quality'] = self.run_evaluation_imp(
                X_train_imps=X_train_imps, X_train_origins=X_train_origins,
                X_train_masks=X_train_masks, seed=seed
            )

        if 'pred_downstream_local' in metrics:
            print("Evaluating downstream prediction...")
            results['pred_downstream_local'] = self.run_evaluation_pred(
                X_train_imps=X_train_imps, X_train_origins=X_train_origins, y_trains=y_trains,
                X_tests=X_tests, y_tests=y_tests, data_config=env.data_config, model = 'nn', seed = seed
            )

        if 'pred_downstream_fed' in metrics:
            print("Evaluating federated downstream prediction...")
            results['pred_downstream_fed'] = self.run_evaluation_fed_pred(
                X_train_imps=X_train_imps, X_train_origins=X_train_origins, y_trains=y_trains,
                X_tests=X_tests, y_tests=y_tests, X_test_global=env.server.X_test_global,
                y_test_global=env.server.y_test_global, data_config=env.data_config, seed=seed
            )

        print("Evaluation completed.")
        self.results = results

        return results

    def save_results(self, results: Dict, save_path: str):
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

    def show_results(self):

        # check empty
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
        else:
            # setup formatting widths
            metrics_w = []
            if 'imp_quality' in self.results:
                metrics = list(self.results['imp_quality']['imp_quality'].keys())
                metrics_w.append([len(m) for m in metrics])
            if 'pred_downstream_local' in self.results:
                metrics = list(self.results['pred_downstream_local']['pred_performance'].keys())
                metrics_w.append([len(m) for m in metrics])
            if 'pred_downstream_fed' in self.results:
                metrics = list(self.results['pred_downstream_fed']['global'].keys())
                metrics_w.append([len(m) for m in metrics])

            metrics_w_array = np.zeros([len(metrics_w), len(max(metrics_w, key=lambda x: len(x)))])
            for i, j in enumerate(metrics_w):
                metrics_w_array[i][0:len(j)] = j

            widths = np.max(metrics_w_array, axis=0).astype(int).tolist()
            total_width = 30 + 3 + sum(widths) + 15 * len(widths) + 5
            print("=" * total_width)
            print("Evaluation Results")
            print("=" * total_width)

            # print results
            if 'imp_quality' in self.results:
                stdout = f"{'Imputation Quality':30} | "
                for idx, (metric, values) in enumerate(self.results['imp_quality']['imp_quality'].items()):
                    mean = np.mean(values)
                    std = np.std(values)
                    stdout += f"{metric:>{widths[idx]}}: {mean:.3f} ({std:.2f}) "
                print(stdout)

            if 'pred_downstream_local' in self.results:
                stdout = f"{'Downstream Prediction (Local)':30} | "
                for idx, (metric, values) in enumerate(self.results['pred_downstream_local']['pred_performance'].items()):
                    mean = np.mean(values)
                    std = np.std(values)
                    stdout += f"{metric:>{widths[idx]}}: {mean:.3f} ({std:.2f}) "
                print(stdout)

            if 'pred_downstream_fed' in self.results:
                stdout = f"{'Downstream Prediction (Fed)':30} | "
                for idx, (metric, values) in enumerate(self.results['pred_downstream_fed']['global'].items()):
                    mean = np.mean(values)
                    std = np.std(values)
                    stdout += f"{metric:>{widths[idx]}}: {mean:.3f} ({std:.2f}) "
                print(stdout)

            print("=" * total_width)

    def run_evaluation_imp(
            self, X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], X_train_masks: List[np.ndarray],
            imp_quality_metrics=None, imp_fairness_metrics=None, seed: int = 0
    ):

        # imputation quality
        if imp_quality_metrics is None:
            imp_quality_metrics = ['rmse', 'nrmse', 'sliced-ws']
        if imp_fairness_metrics is None:
            imp_fairness_metrics = ['variance', 'jain-index']
        imp_qualities = self._evaluate_imp_quality(
            imp_quality_metrics, X_train_imps, X_train_origins, X_train_masks, seed
        )

        # imputation fairness
        imp_fairness = self._evaluation_imp_fairness(imp_fairness_metrics, imp_qualities)

        # clean results
        for key, value in imp_qualities.items():
            imp_qualities[key] = list(value)

        results = {
            'imp_quality': imp_qualities,
            'imp_fairness': imp_fairness,
        }

        return results

    def run_evaluation_pred(
            self, X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], y_trains: List[np.ndarray],
            X_tests: List[np.ndarray], y_tests: List[np.ndarray], data_config: dict,
            model: str = 'nn', model_params=None, pred_fairness_metrics=None,
            seed: int = 0
    ):

        if pred_fairness_metrics is None:
            pred_fairness_metrics = ['variance', 'jain-index']
        if model_params is None:
            model_params = {'weight_decay': 0.0}

        pred_performance = self._evaluation_downstream_prediction(
            model, model_params, X_train_imps, X_train_origins, y_trains,
            X_tests, y_tests, data_config, seed
        )
        pred_performance_fairness = self._evaluation_imp_fairness(pred_fairness_metrics, pred_performance)

        return {
            'pred_performance': pred_performance,
            'pred_performance_fairness': pred_performance_fairness,
        }

    def run_evaluation_fed_pred(
            self, X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], y_trains: List[np.ndarray],
            X_tests: List[np.ndarray], y_tests: List[np.ndarray], X_test_global: np.ndarray, y_test_global: np.ndarray,
            data_config: dict, model_params: dict = None, train_params: dict = None, seed: int = 0
    ):
        if model_params is None:
            model_params = {'batch_norm': True}

        if train_params is None:
            train_params = {
                'global_epoch': 100,
                'local_epoch': 10,
                'fine_tune_epoch': 200,
                'tol': 0.001,
                'patience': 10,
                'batchnorm_avg': True
            }

        pred_performance = self._eval_downstream_fed_prediction(
            model_params, train_params, X_train_imps, X_train_origins, y_trains, X_tests, y_tests,
            X_test_global, y_test_global, data_config, seed
        )

        return pred_performance

    @staticmethod
    def _evaluate_imp_quality(
            metrics: List[str], X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray],
            X_train_masks: List[np.ndarray], seed
    ) -> dict:
        ret_all = {metric: [] for metric in metrics}
        for metric in metrics:
            if metric == 'rmse':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    # print(X_train_imp.shape, X_train_origin.shape, X_train_mask.shape)
                    ret.append(rmse(X_train_imp, X_train_origin, X_train_mask))
            elif metric == 'nrmse':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    rmse_ = np.sqrt(np.mean((X_train_imp - X_train_origin) ** 2))
                    std = np.std(X_train_origin)
                    ret.append(rmse_ / std)
            elif metric == 'mae':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    ret.append(np.mean(np.abs(X_train_imp - X_train_origin)))
            elif metric == 'nmae':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    mae_ = np.mean(np.abs(X_train_imp - X_train_origin))
                    std = np.std(X_train_origin)
                    ret.append(mae_ / std)
            elif metric == 'sliced-ws':
                ret = []
                for X_train_imp, X_train_origin in zip(X_train_imps, X_train_origins):
                    ret.append(sliced_ws(X_train_imp, X_train_origin, N=100, seed=seed))
            else:
                raise ValueError(f"Invalid metric: {metric}")

            ret_all[metric] = ret

        return ret_all

    @staticmethod
    def _evaluation_imp_fairness(metrics, imp_qualities: Dict[str, List[float]]) -> dict:

        ret = {metric: {} for metric in metrics}
        for metric in metrics:
            for quality_metric, imp_quality in imp_qualities.items():
                if metric == 'variance':
                    ret[metric][quality_metric] = np.std(imp_quality)
                elif metric == 'cosine-similarity':
                    imp_quality = np.array(imp_quality)
                    ret[metric][quality_metric] = np.dot(imp_quality, imp_quality) / (np.linalg.norm(imp_quality) ** 2)
                elif metric == 'jain-index':
                    ret[metric][quality_metric] = np.sum(imp_quality) ** 2 / (
                            len(imp_quality) * np.sum([x ** 2 for x in imp_quality]))
                elif metric == 'entropy':
                    imp_quality = np.array(imp_quality)
                    imp_quality = imp_quality / np.sum(imp_quality)
                    ret[metric][quality_metric] = -np.sum(imp_quality * np.log(imp_quality))
                else:
                    raise ValueError(f"Invalid metric: {metric}")

        return ret

    @staticmethod
    def _evaluation_downstream_prediction(
            model: str, model_params: dict,
            X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], y_trains: List[np.ndarray],
            X_tests: List[np.ndarray], y_tests: List[np.ndarray], data_config: dict, seed: int = 0
    ):

        try:
            task_type = data_config['task_type']
            clf_type = data_config['clf_type']
        except KeyError:
            raise KeyError("task_type and clf_type is not defined in data_config")

        assert task_type in ['classification', 'regression'], f"Invalid task_type: {task_type}"
        if task_type == 'classification':
            assert clf_type in ['binary-class', 'multi-class', 'binary'], f"Invalid clf_type: {clf_type}"

        ################################################################################################################
        # Loader classification model
        if model == 'linear':
            if task_type == 'classification':
                clf = LogisticRegressionCV(
                    Cs=5, class_weight='balanced', solver='saga', random_state=seed, max_iter=1000, **model_params
                )
            else:
                clf = RidgeCV(alphas=[1], **model_params)
                # clf = LinearRegression(**model_params)
        elif model == 'tree':
            if task_type == 'classification':
                clf = RandomForestClassifier(
                    n_estimators=100, class_weight='balanced', random_state=seed, **model_params
                )
            else:
                clf = RandomForestRegressor(n_estimators=100, random_state=seed, **model_params)
        elif model == 'nn':
            set_seed(seed)
            if task_type == 'classification':
                clf = TwoNNClassifier(**model_params)
            else:
                clf = TwoNNRegressor(**model_params)
        else:
            raise ValueError(f"Invalid model: {model}")

        models = [deepcopy(clf) for _ in range(len(X_train_imps))]
        ################################################################################################################
        # Evaluation
        if task_type == 'classification':
            eval_metrics = ['accu', 'f1', 'auc', 'prc']
        else:
            eval_metrics = ['mse', 'mae', 'msle']

        ret = {eval_metric: [] for eval_metric in eval_metrics}
        y_min = np.concatenate(y_trains).min()
        y_max = np.concatenate(y_trains).max()
        for idx in trange(len(X_train_imps), desc='Clients', leave=False, colour='blue'):

            X_train_imp = X_train_imps[idx]
            y_train = y_trains[idx]
            X_test = X_tests[idx]
            y_test = y_tests[idx]
            clf = models[idx]
            clf.fit(X_train_imp, y_train)
            y_pred = clf.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = clf.predict_proba(X_test)
            else:
                y_pred[y_pred < y_min] = y_min
                y_pred[y_pred > y_max] = y_max
                y_pred_proba = None

            for eval_metric in eval_metrics:
                ret[eval_metric].append(task_eval(
                    eval_metric, task_type, clf_type, y_pred, y_test, y_pred_proba
                ))

        return ret

    @staticmethod
    def _eval_downstream_fed_prediction(
            model_params: dict, train_params: dict,
            X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], y_trains: List[np.ndarray],
            X_tests: List[np.ndarray], y_tests: List[np.ndarray], X_test_global, y_test_global,
            data_config: dict, seed: int = 0
    ):

        # Federated Prediction
        global_epoch = train_params['global_epoch']
        local_epoch = train_params['local_epoch']
        fine_tune_epoch = train_params['fine_tune_epoch']
        batchnorm_avg = train_params['batchnorm_avg']
        tol = train_params['tol']
        patience = train_params['patience']

        try:
            task_type = data_config['task_type']
            clf_type = data_config['clf_type']
        except KeyError:
            raise KeyError("task_type is not defined in data_config")

        assert task_type in ['classification', 'regression'], f"Invalid task_type: {task_type}"
        if task_type == 'classification':
            assert clf_type in ['binary-class', 'multi-class', 'binary'], f"Invalid clf_type: {clf_type}"

        ################################################################################################################
        # Loader classification model
        set_seed(seed)
        if task_type == 'classification':
            clf = TwoNNClassifier(optimizer='sgd', epochs=local_epoch, **model_params)
        else:
            clf = TwoNNRegressor(optimizer='sgd', epochs=local_epoch, **model_params)

        ################################################################################################################
        # Evaluation
        if task_type == 'classification':
            eval_metrics = ['accuracy', 'f1', 'auc', 'prc']
        else:
            eval_metrics = ['mse', 'mae', 'msle']

        models = [deepcopy(clf) for _ in range(len(X_train_imps))]
        weights = [len(X_train_imp) for X_train_imp in X_train_imps]
        weights = [weight / sum(weights) for weight in weights]
        early_stoppings = [
            EarlyStopping(
                tolerance=tol, tolerance_patience=patience, increase_patience=patience,
                window_size=1, check_steps=1, backward_window_size=1) for _ in range(len(X_train_imps))
        ]
        early_stopping_signs = [False for _ in range(len(X_train_imps))]

        ################################################################################################################
        # Training
        for epoch in trange(global_epoch, desc='Global Epoch', leave=False, colour='blue'):
            ############################################################################################################
            # Local training
            losses = {}
            for idx, (X_train_imp, X_train_origin, y_train, clf) in enumerate(zip(
                    X_train_imps, X_train_origins, y_trains, models
            )):
                if early_stopping_signs[idx]:
                    continue
                ret = clf.fit(X_train_imp, y_train)
                losses[idx] = ret['loss']

            ############################################################################################################
            # Server aggregation the parameters of local models of clients (pytorch model)
            aggregated_state_dict = OrderedDict()

            for idx, model in enumerate(models):
                local_state_dict = model.get_parameters()
                for key, param in local_state_dict.items():
                    if batchnorm_avg:
                        if key in aggregated_state_dict:
                            aggregated_state_dict[key] += param * weights[idx]
                        else:
                            aggregated_state_dict[key] = param * weights[idx]
                    else:
                        if key in ['running_mean', 'running_var', 'num_batches_tracked']:
                            continue
                        if key in aggregated_state_dict:
                            aggregated_state_dict[key] += param * weights[idx]
                        else:
                            aggregated_state_dict[key] = param * weights[idx]

            ############################################################################################################
            # local update
            for idx, model in enumerate(models):
                if early_stopping_signs[idx]:
                    continue
                model.update_parameters(aggregated_state_dict.copy())

            # early stopping
            for idx, model in enumerate(models):
                if early_stopping_signs[idx]:
                    continue
                early_stoppings[idx].update(losses[idx])
                if early_stoppings[idx].check_convergence():
                    loguru.logger.debug(f"Early stopping at epoch {epoch}")
                    early_stopping_signs[idx] = True

            if all(early_stopping_signs):
                break

        ################################################################################################################
        # prediction and evaluation
        local_ret = {eval_metric: [] for eval_metric in eval_metrics}
        y_min = np.concatenate(y_trains).min()
        y_max = np.concatenate(y_trains).max()
        for idx, (X_train_imp, X_train_origin, y_train, X_test, y_test) in enumerate(zip(
                X_train_imps, X_train_origins, y_trains, X_tests, y_tests
        )):
            y_pred = clf.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = clf.predict_proba(X_test)
            else:
                y_pred_proba = None
                y_pred[y_pred < y_min] = y_min
                y_pred[y_pred > y_max] = y_max

            for eval_metric in eval_metrics:
                local_ret[eval_metric].append(task_eval(
                    eval_metric, task_type, clf_type, y_pred, y_test, y_pred_proba
                ))

        y_pred_global = clf.predict(X_test_global)
        if task_type == 'classification':
            y_pred_proba_global = clf.predict_proba(X_test_global)
        else:
            y_pred_proba_global = None

        global_ret = {}
        for eval_metric in eval_metrics:
            if eval_metric not in global_ret:
                global_ret[eval_metric] = []
            global_ret[eval_metric].append(task_eval(
                eval_metric, task_type, clf_type, y_pred_global, y_test_global, y_pred_proba_global
            ))

        ################################################################################################################
        # fine-tuning
        for idx, (X_train_imp, X_train_origin, y_train, clf) in enumerate(
                zip(X_train_imps, X_train_origins, y_trains, models)
        ):
            clf.epochs = fine_tune_epoch
            clf.fit(X_train_imp, y_train)

        ret_personalized = {eval_metric: [] for eval_metric in eval_metrics}
        for idx, (X_train_imp, X_train_origin, y_train, X_test, y_test, clf) in enumerate(zip(
                X_train_imps, X_train_origins, y_trains, X_tests, y_tests, models
        )):
            y_pred = clf.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = clf.predict_proba(X_test)
            else:
                y_pred_proba = None

            for eval_metric in eval_metrics:
                ret_personalized[eval_metric].append(task_eval(
                    eval_metric, task_type, clf_type, y_pred, y_test, y_pred_proba
                ))

        return {
            'global': global_ret,
            'local': local_ret,
            'personalized': ret_personalized
        }
