import json
from collections import OrderedDict
from copy import deepcopy
from typing import List, Dict, Union
import warnings
import os
import loguru
import numpy as np
import sys

from sklearn.manifold import TSNE
import gower
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from typing import TYPE_CHECKING
from typing import Tuple
if TYPE_CHECKING:
    from fedimpute.execution_environment import FedImputeEnv

warnings.filterwarnings("ignore")
from tqdm.auto import trange
from tabulate import tabulate

from .imp_quality_metrics import rmse, sliced_ws
from .twonn import TwoNNRegressor, TwoNNClassifier
from .pred_model_metrics import task_eval
from ..utils.reproduce_utils import set_seed
from ..utils.nn_utils import EarlyStopping
from ..utils.logger import setup_logger
from .fed_prediction import (
    eval_fed_pred_torchnn, 
    eval_fed_pred_sklnn, 
    eval_fed_pred_lr, 
    eval_fed_pred_svm, 
    eval_fed_pred_xgboost, 
    eval_fed_pred_rf
)
from .fed_regression_analysis import (
    eval_fed_reg_logit, 
    eval_fed_reg_linear
)


class Evaluator:

    """
    Evaluator class for the federated imputation environment
    """

    def __init__(
        self,
        debug: bool = False
    ):
        self.results = None
        self.debug = debug
        
        if debug is False:
            loguru.logger.remove()
            loguru.logger.add(
                sys.stdout, format="<level>{message}</level>", level="INFO"
            )
        else:
            loguru.logger.add(sys.stderr, level="DEBUG")
    
    def save_results(
        self, 
        results: Dict, 
        save_path: str
    ):
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

    def evaluate_all(
        self, 
        env: 'FedImputeEnv', 
        metrics: Union[List, None] = None,
        model: str = 'rf', 
        seed: int = 0,
        verbose: int = 1
    ):
        
        if metrics is None:
            metrics = ['imp_quality', 'local_pred', 'fed_pred']

        for metric in metrics:
            if metric not in ['imp_quality', 'local_pred', 'fed_pred']:
                raise ValueError(f"Invalid metric: {metric}")

        results = {}
        
        X_train_origins, y_trains = env.get_data(client_ids='all', data_type = 'train', include_y=True)
        X_tests, y_tests = env.get_data(client_ids='all', data_type = 'test', include_y=True)
        X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
        X_test_imps = env.get_data(client_ids='all', data_type = 'test_imp')
        X_train_masks = env.get_data(client_ids='all', data_type = 'train_mask')
        X_test_masks = env.get_data(client_ids='all', data_type = 'test_mask')
        X_global_test_imp = env.get_data(data_type = 'global_test_imp')
        y_global_test = env.get_data(data_type = 'y_global_test')
        data_config = env.get_data(data_type = 'config')

        if 'imp_quality' in metrics:
            if verbose >= 1:
                loguru.logger.info("Evaluating imputation quality...")
            ret = self.evaluate_imp_quality(
                X_train_imps=X_train_imps, X_train_origins=X_train_origins,
                X_train_masks=X_train_masks, seed=seed, verbose = verbose
            )
            
            results['imp_quality'] = ret['imp_quality']
            
            if verbose >= 1:
                loguru.logger.info("Imputation quality evaluation completed.")

        if 'local_pred' in metrics:
            if verbose >= 1:
                loguru.logger.info("Evaluating downstream prediction...")
            ret = self.run_local_prediction(
                X_train_imps=X_train_imps, y_trains=y_trains, X_tests=X_test_imps, y_tests=y_tests, 
                data_config=data_config, model = model, seed = seed, verbose = verbose
            )
            results['local_pred'] = ret['local_pred']
            results['local_pred_fairness'] = ret['local_pred_fairness']
            if verbose >= 1:
                loguru.logger.info("Downstream prediction evaluation completed.")

        if 'fed_pred' in metrics:
            if verbose >= 1:
                loguru.logger.info("Evaluating federated downstream prediction...")
            ret = self.run_fed_prediction(
                X_train_imps=X_train_imps, y_trains=y_trains, X_tests=X_test_imps, y_tests=y_tests, 
                X_test_global=X_global_test_imp, y_test_global=y_global_test, 
                data_config=data_config, model_name=model, n_rounds=3, seed=seed, verbose = verbose
            )
            results['fed_pred'] = ret['fed_pred']
            if verbose >= 1:
                loguru.logger.info("Federated downstream prediction evaluation completed.")

        if verbose >= 1:
            loguru.logger.info("Evaluation completed.")
        
        self.results = results

        return results

    def show_results_all(self, format: str = 'dataframe'):

        # check empty
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
        else:
            if format == 'dataframe':
                df = self.export_results(format = 'dataframe')
                # format the dataframe
                df = df.map(lambda x: f"{x:.3f}")
                return df
            elif format == 'text-table':
                print(self.export_results(format = 'text-table'))
            else:
                raise ValueError(f"Invalid format: {format}")
                
    def evaluate_imp_quality(
        self, 
        X_train_imps: List[pd.DataFrame], 
        X_train_origins: List[pd.DataFrame], 
        X_train_masks: List[pd.DataFrame],
        metrics=None, 
        seed: int = 0,
        verbose: int = 0
    ):
        setup_logger(verbose)

        X_train_imps = [item.values for item in X_train_imps]
        X_train_origins = [item.values for item in X_train_origins]
        X_train_masks = [item.values for item in X_train_masks]

        # imputation quality
        if metrics is None:
            metrics = ['rmse', 'nrmse', 'sliced-ws']
        imp_qualities = self._evaluate_imp_quality(
            metrics, X_train_imps, X_train_origins, X_train_masks, seed
        )

        # clean results
        for key, value in imp_qualities.items():
            imp_qualities[key] = list(value)

        results = imp_qualities
        
        if self.results is None:
            self.results = {}
        
        self.results['imp_quality'] = results

        return {
            'imp_quality': results,
        }
    
    def show_imp_results(self):
        """
        Example:
        {
            'imp_quality': {
                'rmse': [0.1, 0.1, 0.1],
                'nrmse': [0.2, 0.2, 0.2],
                'sliced-ws': [0.3, 0.3, 0.3]
            }
        }
        """
        
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
            return

        total_width = 48
        print("=" * total_width)
        print("Imputation Quality")
        print("=" * total_width)
        metrics = list(self.results['imp_quality'].keys())
        num_clients = len(list(self.results['imp_quality'].values())[0])
        ret = self.results['imp_quality']
        headers = [""] + metrics
        rows = []
        
        # Add client rows
        for i in range(num_clients):
            client_row = [f"Client {i+1}"]
            for metric in metrics:
                values = ret[metric]
                client_row.append(f"{values[i]:.3f}")
            rows.append(client_row)
            
        # Add separator
        rows.append(["-" * 10] * (len(metrics) + 1))
        
        # Average row
        averages = ["Average"]
        for metric in metrics:
            values = ret[metric]
            averages.append(f"{np.mean(values):.3f}")
        rows.append(averages)
        
        # Std row
        stds = ["Std"]
        for metric in metrics:
            values = ret[metric]
            stds.append(f"{np.std(values):.3f}")
        rows.append(stds)
        
        # Print with red dashed lines as separators
        print(tabulate(rows, headers=headers, stralign="center", numalign="center"))
        print('=' * total_width)
        
    def tsne_visualization(
        self, 
        X_imps: List[pd.DataFrame], 
        X_origins: List[pd.DataFrame], 
        fontsize: int = 20,
        alpha: float = 0.5,
        sampling_size: int = None,
        overall: bool = False,
        seed: int = 0,
        save_path: str = None
    ):

        X_imps = [item.values for item in X_imps]
        X_origins = [item.values for item in X_origins]

        color_mapping = {
            'original': 'red',
            'imputed': 'blue'
        }

        def eval_tsne(origin_data, imputed_data, save_path: str = None):

            # overall
            plot_data = np.concatenate((origin_data, imputed_data), axis=0)
            N1 = origin_data.shape[0]
            N2 = imputed_data.shape[0]
            colors = [color_mapping['original'] for i in range(N1)] + [color_mapping['imputed'] for i in range(N2)]
            tsne = TSNE(
                metric='precomputed', n_components=2, verbose=0, n_iter=1000, perplexity=40,
                n_iter_without_progress=300, init='random', n_jobs=-1, random_state=seed
            )

            tsne_results = tsne.fit_transform(np.clip(gower.gower_matrix(plot_data), 0, 1))

            return tsne_results, colors, N1, N2

        def plot_tsne(tsne_results, colors, N1, N2, alpha = 0.5, ax = None):
            ax.scatter(tsne_results[:N1, 0], tsne_results[:N1, 1], c=color_mapping['original'], label='original', alpha = alpha)
            ax.scatter(tsne_results[N1:, 0], tsne_results[N1:, 1], c=color_mapping['imputed'], label='imputed', alpha = alpha)
            return ax
        
        if overall:
            X_imp = np.concatenate(X_imps, axis=0)
            X_origin = np.concatenate(X_origins, axis=0)
            X_imps.append(X_imp)
            X_origins.append(X_origin)
            
            titles = [f"Client {i+1}" for i in range(len(X_imps))]
            titles[-1] = 'Overall'
        else:
            titles = [f"Client {i+1}" for i in range(len(X_imps))]
        
        n_clients = len(X_imps)
        
        if sampling_size is not None:
            np.random.seed(seed)
            for i in range(n_clients):
                indices = np.random.choice(len(X_imps[i]), sampling_size, replace=False)
                X_imps[i] = X_imps[i][indices]
                X_origins[i] = X_origins[i][indices]
        
        n_cols = 5
        if n_clients < 5:
            n_cols = n_clients
        n_rows = n_clients // n_cols + (n_clients % n_cols > 0)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axs = axs.flatten()
        
        for i in range(n_clients):
            print(f'Evaluating TSNE for {titles[i]} ...')
            tsne_results, colors, N1, N2 = eval_tsne(X_origins[i], X_imps[i])
            plot_tsne(tsne_results, colors, N1, N2, ax=axs[i])
            axs[i].set_title(titles[i], fontsize=fontsize, fontweight='bold')
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        
        for i in range(n_clients, len(axs)):
            axs[i].set_visible(False)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping['original'], markersize=fontsize-3),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping['imputed'], markersize=fontsize-3)
        ]
        
        # add legend to bottom of the plot with out border
        fig.legend(
            legend_elements, 
            ['Original', 'Imputed'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), 
            prop={'weight': 'bold', 'size': fontsize}, frameon=False
        )
        plt.subplots_adjust(wspace=0.0)
        plt.tight_layout()
        if save_path is not None:
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def run_local_regression_analysis(
        self, 
        X_train_imps: List[pd.DataFrame], 
        y_trains: List[pd.Series], 
        data_config: dict,
        verbose: int = 0
    ):
        setup_logger(verbose)
        np.random.seed(seed=233423)
        
        if data_config['task_type'] == 'regression':
            import statsmodels.api as sm
            results = []
            for X_train_imp, y_train in zip(X_train_imps, y_trains):
                X_train_imp = sm.add_constant(X_train_imp)  # Add intercept term
                model = sm.OLS(y_train, X_train_imp)
                fit = model.fit()
                results.append(fit)
        elif data_config['task_type'] == 'classification':
            import statsmodels.api as sm
            results = []
            for X_train_imp, y_train in zip(X_train_imps, y_trains):
                n_classes = len(np.unique(y_train.values))
                if n_classes == 2:
                    X_train_imp = sm.add_constant(X_train_imp)  # Add intercept term
                    model = sm.Logit(y_train, X_train_imp)
                    fit = model.fit(disp=0)  # Suppress convergence messages
                    results.append(fit)
                elif n_classes > 2:
                    X_train_imp = sm.add_constant(X_train_imp)  # Add intercept term
                    model = sm.Logit(y_train, X_train_imp)
                    fit = model.fit(disp=0)  # Suppress convergence messages
                    results.append(fit)
        else:
            raise ValueError(f"Invalid task type: {data_config['task_type']}")
        
        # save results to class attribute
        if self.results is None:
            self.results = {}
        self.results['local_regression'] = results

    def show_local_regression_results(self, client_idx: int):
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
            return

        if client_idx is None:
            client_idx = 0

        if 'local_regression' not in self.results:
            print("Local regression results are not available. Run run_local_regression_analysis first.")
            return

        results = self.results['local_regression']
        result = results[client_idx]
        title = result.summary().tables[0].title
        title  = 'Local ' + title + ' (client ' + str(client_idx + 1) + ')'
        summary = result.summary()
        summary.tables[0].title = title
        print(summary)

    def run_local_prediction(
        self, 
        X_train_imps: List[pd.DataFrame], 
        y_trains: List[pd.Series],
        X_tests: List[pd.DataFrame], 
        y_tests: List[pd.Series], 
        data_config: dict,
        model: str = 'nn', 
        model_params=None, 
        pred_fairness_metrics=None,
        seed: int = 0,
        verbose: int = 0
    ):
        X_train_imps = [item.values for item in X_train_imps]
        X_tests = [item.values for item in X_tests]
        y_trains = [item.values for item in y_trains]
        y_tests = [item.values for item in y_tests]

        setup_logger(verbose)
        if data_config['task_type'] == 'classification':
            y_train_total = np.concatenate(y_trains)
            y_test_total = np.concatenate(y_tests)
            y_total = np.concatenate([y_train_total, y_test_total])
            n_classes = len(np.unique(y_total))
            if n_classes > 2:
                data_config['clf_type'] = 'multi-class'
            else:
                data_config['clf_type'] = 'binary-class'
        else:
            data_config['clf_type'] = None
        

        if pred_fairness_metrics is None:
            pred_fairness_metrics = ['variance', 'jain-index']
        # if model_params is None:
        #     model_params = {'weight_decay': 0.0}

        pred_performance = self._evaluation_downstream_prediction(
            model, model_params, X_train_imps, y_trains,
            X_tests, y_tests, data_config, seed, verbose
        )
        pred_performance_fairness = self._evaluation_imp_fairness(pred_fairness_metrics, pred_performance)
        
        if self.results is None:
            self.results = {}
        
        self.results['local_pred'] = pred_performance
        self.results['local_pred_fairness'] = pred_performance_fairness

        return {
            'local_pred': pred_performance,
            'local_pred_fairness': pred_performance_fairness,
        }
        
    def show_local_prediction_results(self):
        """
        Example:
        {
            'pred_downstream_local': {
                'accuracy': [0.8, 0.8, 0.8],
                'f1': [0.7, 0.7, 0.7],
                'auc': [0.6, 0.6, 0.6],
                'prc': [0.5, 0.5, 0.5]
            }
        }
        """
        
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
            return

        total_width = 58
        print("=" * total_width)
        print("Downstream Prediction (Local)")
        print("=" * total_width)
        metrics = list(self.results['local_pred'].keys())
        num_clients = len(list(self.results['local_pred'].values())[0])
        ret = self.results['local_pred']
        headers = [""] + metrics
        rows = []
        
        # Add client rows
        for i in range(num_clients):
            client_row = [f"Client {i+1}"]
            for metric in metrics:
                values = ret[metric]
                client_row.append(f"{values[i]:.3f}")
            rows.append(client_row)
            
        # Add separator
        rows.append(["-" * 10] * (len(metrics) + 1))
        
        # Average row
        averages = ["Average"]
        for metric in metrics:
            values = ret[metric]
            averages.append(f"{np.mean(values):.3f}")
        rows.append(averages)
        
        # Std row
        stds = ["Std"]
        for metric in metrics:
            values = ret[metric]
            stds.append(f"{np.std(values):.3f}")
        rows.append(stds)
        
        # Print with red dashed lines as separators
        print(tabulate(rows, headers=headers, stralign="center", numalign="center"))
        print('=' * total_width)

    def run_fed_regression_analysis(
        self, 
        X_train_imps: List[pd.DataFrame], 
        y_trains: List[pd.Series],
        data_config: dict,
        verbose: int = 0
    ):

        setup_logger(verbose)

        if data_config['task_type'] == 'regression':
            result = eval_fed_reg_linear(X_train_imps, y_trains)
            title = 'Federated Linear Regression Result'
        elif data_config['task_type'] == 'classification':
            if len(np.unique(np.concatenate([item.values for item in y_trains], axis=0))) == 2:
                result = eval_fed_reg_logit(X_train_imps, y_trains)
                title = 'Federated Logit Regression Result'
            else:
                raise NotImplementedError("Multi-class federated logit regression is not implemented yet.")
        else:
            raise ValueError(f"Invalid task type: {data_config['task_type']}")
        
        if self.results is None:
            self.results = {}
        self.results['fed_regression'] = {
            'result': result,
            'title': title
        }
    
    def show_fed_regression_results(self):
        if self.results is None or len(self.results) == 0:
            print("Evaluation results is empty. Run evaluation first.")
            return
        summary = self.results['fed_regression']['result'].summary()
        summary.tables[0].title = self.results['fed_regression']['title']
        print(summary)

    def run_fed_prediction(
        self, 
        X_train_imps: List[pd.DataFrame], 
        y_trains: List[pd.Series],
        X_tests: List[pd.DataFrame], 
        y_tests: List[pd.Series], 
        X_test_global: pd.DataFrame, 
        y_test_global: pd.Series,
        data_config: dict, 
        model_name: str,
        model_params: dict = None, 
        train_params: dict = None,
        n_rounds: int = 1, 
        seed: int = 0,
        verbose: int = 0
    ):
        X_train_imps = [item.values for item in X_train_imps]
        X_tests = [item.values for item in X_tests]
        y_trains = [item.values for item in y_trains]
        y_tests = [item.values for item in y_tests]
        X_test_global = X_test_global.values
        y_test_global = y_test_global.values

        setup_logger(verbose)

        if model_params is None:
            model_params = {}
        if train_params is None:
            train_params = {}

        if data_config['task_type'] == 'classification':
            y_train_total = np.concatenate(y_trains)
            y_test_total = np.concatenate(y_tests)
            y_total = np.concatenate([y_train_total, y_test_total, y_test_global])
            n_classes = len(np.unique(y_total))
            if n_classes > 2:
                data_config['clf_type'] = 'multi-class'
            else:
                data_config['clf_type'] = 'binary-class'
        else:
            data_config['clf_type'] = None

        if model_name == 'torchnn':
            eval_func = eval_fed_pred_torchnn
        elif model_name == 'sklnn':
            eval_func = eval_fed_pred_sklnn
        elif model_name == 'lr':
            eval_func = eval_fed_pred_lr
        elif model_name == 'svm':
            eval_func = eval_fed_pred_svm
        elif model_name == 'xgboost':
            eval_func = eval_fed_pred_xgboost
        elif model_name == 'rf':
            eval_func = eval_fed_pred_rf
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        if self.results is None:
                self.results = {}

        pred_performances = []
        for i in range(n_rounds):
            seed = seed + i
            pred_performance = eval_func(
                model_params, train_params, 
                X_train_imps, y_trains, X_tests, y_tests, X_test_global, y_test_global, 
                data_config, seed, verbose
            )

            pred_performances.append(pred_performance)

        # Merge results from multiple rounds
        merged_results = {
            'global': {},
            'personalized': {}
        }
            
        # Get all metrics from the first round
        global_metrics = pred_performances[0]['global'].keys()
        personalized_metrics = pred_performances[0]['personalized'].keys()
            
        # Average global metrics across rounds
        for metric in global_metrics:
            values = [round_result['global'][metric] for round_result in pred_performances]
            merged_results['global'][metric] = np.mean(values)
            
        # Average personalized metrics across rounds
        for metric in personalized_metrics:
            
            num_clients = len(pred_performances[0]['personalized'][metric])
            client_values = [[] for _ in range(num_clients)]
            
            for round_result in pred_performances:
                for client_idx, client_value in enumerate(round_result['personalized'][metric]):
                    client_values[client_idx].append(client_value)
            
            merged_results['personalized'][metric] = [np.mean(client) for client in client_values]
        
        self.results['fed_pred'] = merged_results

        return {
            'fed_pred': merged_results,
        }

    def show_fed_prediction_results(self):
        
        """
        Example:
        {
            'fed_pred': {
                'global': {'accuracy': 0.8, 'f1': 0.8, 'auc': 0.8, 'prc': 0.8},
                'personalized': {
                    'accuracy': [0.7, 0.7, 0.7],
                    'f1': [0.6, 0.6, 0.6],
                    'auc': [0.5, 0.5, 0.5],
                    'prc': [0.4, 0.4, 0.4]
                }, 
            }
        }
        {
            'fed_pred': {
                'global': {'accuracy': 0.8, 'f1': 0.8, 'auc': 0.8, 'prc': 0.8},
                'personalized': {
                    'accuracy': [0.7, 0.7, 0.7],
                    'f1': [0.6, 0.6, 0.6],
                    'auc': [0.5, 0.5, 0.5],
                    'prc': [0.4, 0.4, 0.4]
                }
            }
        }
        """

        if self.results is None or len(self.results) == 0 or 'fed_pred' not in self.results:
            print("Evaluation federated prediction results is empty. Run evaluation first.")
            return        
        
        total_width = 63
        print("=" * total_width)
        print("Downstream Prediction (Fed)")
        print("=" * total_width)
        metrics = list(self.results['fed_pred']['global'].keys())
        num_clients = len(list(self.results['fed_pred']['personalized'].values())[0])
        ret = self.results['fed_pred']['personalized']
        ret_global = self.results['fed_pred']['global']
            
        headers = ["Personalized"] + metrics
        rows = []
        
        # Add client rows
        for i in range(num_clients):
            client_row = [f"Client {i+1}"]
            for metric in metrics:
                values = ret[metric]
                client_row.append(f"{values[i]:.3f}")
            rows.append(client_row)
            
        # Add separator
        rows.append(["-" * 10] * (len(metrics) + 1))
        
        # Global FL
        averages = ["Global"]
        for metric in metrics:
            values = ret_global[metric]
            averages.append(f"{np.mean(values):.3f}")
        rows.append(averages)
    
        # Print with red dashed lines as separators
        print(tabulate(rows, headers=headers, stralign="center", numalign="center"))
        print('=' * total_width)

    @staticmethod
    def _evaluate_imp_quality(
        metrics: List[str], 
        X_train_imps: List[np.ndarray], 
        X_train_origins: List[np.ndarray],
        X_train_masks: List[np.ndarray], 
        seed: int = 0
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
    def _evaluation_imp_fairness(
        metrics, 
        imp_qualities: Dict[str, List[float]]
    ) -> dict:

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
        model: str, 
        model_params: dict,
        X_train_imps: List[np.ndarray], 
        y_trains: List[np.ndarray],
        X_tests: List[np.ndarray], 
        y_tests: List[np.ndarray], 
        data_config: dict, 
        seed: int = 0,
        verbose: int = 1
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
        if model == 'lr':
            if task_type == 'classification':
                if model_params is None or len(model_params.keys()) == 0:
                    model_params = {
                        'C': 100,
                        'penalty': 'l2',
                        'class_weight': 'balanced',
                        'solver': 'newton-cg',
                        'max_iter': 1000,
                        'random_state': seed,
                    }
                clf = LogisticRegression(
                    **model_params
                )
            else:
                clf = RidgeCV()
        elif model == 'rf':
            if model_params is None or len(model_params.keys()) == 0:
                model_params = {
                    'n_estimators': 200,
                    'random_state': seed,
                    'max_depth': 10,
                    'min_samples_leaf': 2,
                    'max_features': None,
                }
            if task_type == 'classification':
                clf = RandomForestClassifier(class_weight='balanced', **model_params)
            else:
                clf = RandomForestRegressor(**model_params)
        elif model == 'torchnn':
            set_seed(seed)
            if task_type == 'classification':
                clf = TwoNNClassifier(**model_params)
            else:
                clf = TwoNNRegressor(**model_params)
        elif model == 'sklnn':
            if task_type == 'classification':
                clf = MLPClassifier(
                    hidden_layer_sizes=(100, 100),
                    max_iter=1000,
                    random_state=seed,
                    **model_params
                )
            else:
                clf = MLPRegressor(
                    hidden_layer_sizes=(100, 100),
                    max_iter=1000,
                    random_state=seed, **model_params)
        else:
            raise ValueError(f"Invalid model: {model}")

        models = [deepcopy(clf) for _ in range(len(X_train_imps))]
        ################################################################################################################
        # Evaluation
        if task_type == 'classification':
            eval_metrics = ['accuracy', 'f1', 'auc', 'prc']
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
            
            if verbose >= 1:
                loguru.logger.debug(f"Prediction completed for client {idx}.")

        return ret

    @staticmethod
    def _eval_downstream_fed_prediction(
        model_params: dict, 
        train_params: dict,
        X_train_imps: List[np.ndarray], 
        y_trains: List[np.ndarray],
        X_tests: List[np.ndarray], 
        y_tests: List[np.ndarray], 
        X_test_global, 
        y_test_global,
        data_config: dict, 
        seed: int = 0,
        verbose: int = 1
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
            for idx, (X_train_imp, y_train, clf) in enumerate(zip(X_train_imps, y_trains, models)):
                if early_stopping_signs[idx]:
                    continue
                ret = clf.fit(X_train_imp, y_train)
                losses[idx] = ret['loss']
            
            if verbose >= 1:
                if epoch % (global_epoch // 10) == 0:
                    loguru.logger.info(f"Epoch {epoch} - average loss: {np.mean(list(losses.values()))}")

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
                    if verbose >= 1:
                        loguru.logger.info(f"Early stopping at epoch {epoch}")
                    early_stopping_signs[idx] = True

            if all(early_stopping_signs):
                break

        ################################################################################################################
        # prediction and evaluation
        local_ret = {eval_metric: [] for eval_metric in eval_metrics}
        y_min = np.concatenate(y_trains).min()
        y_max = np.concatenate(y_trains).max()
        for idx, (X_train_imp, y_train, X_test, y_test) in enumerate(zip(X_train_imps, y_trains, X_tests, y_tests)):
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
        for idx, (X_train_imp, y_train, clf) in enumerate(zip(X_train_imps, y_trains, models)):
            clf.epochs = fine_tune_epoch
            clf.fit(X_train_imp, y_train)

        ret_personalized = {eval_metric: [] for eval_metric in eval_metrics}
        for idx, (X_train_imp, y_train, X_test, y_test, clf) in enumerate(zip(
                X_train_imps, y_trains, X_tests, y_tests, models
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
        
    def export_results(self, format: str = 'dataframe', persist: bool = False, save_path: str = None):
        """
        Export the results to a dataframe or a dictionary.
        
        Parameters:
            format (str): The format of the output. Can be 'dataframe', 'dict-dataframe', 'dict', or 'text-table'.
            persist (bool): If True, the results will be saved to a file. [TODO]
            save_path (str): The path to save the results. [TODO]
        
        Returns:
            ret (pd.DataFrame or dict or str): The results in the specified format.
        """
        if self.results is None:
            raise ValueError("No results to export, please run the evaluation first.")
        
        ################################################################################################################
        # Export to one Pandas Dataframe
        if format == 'dataframe':
            df = pd.DataFrame()
            if 'imp_quality' in self.results:
                for metric_name, metric_values in self.results['imp_quality'].items():
                    col_name = ('imp_quality', metric_name)
                    df[col_name] = metric_values
            
            if 'local_pred' in self.results:
                for metric_name, metric_values in self.results['local_pred'].items():
                    col_name = ('local_pred', f'local_{metric_name}')
                    df[col_name] = metric_values
                    
            if 'fed_pred' in self.results:
                for metric_name, metric_values in self.results['fed_pred']['personalized'].items():
                    col_name = ('fed_pred_personalized', f'personalized_{metric_name}')
                    df[col_name] = metric_values
                
                num_clients = len(list(self.results['fed_pred']['personalized'].values())[0])
                for metric_name, metric_values in self.results['fed_pred']['global'].items():
                    col_name = ('fed_pred_global', f'global_{metric_name}')
                    df[col_name] = [metric_values[0]]*num_clients
            
            # covert tuple columns to multiindex columns
            df.columns = pd.MultiIndex.from_tuples(df.columns)
                 
            return df
        
        ################################################################################################################
        # Export to dictionary of dataframes
        elif format == 'dict-dataframe':
            ret = {}
            if 'imp_quality' in self.results:
                df = pd.DataFrame()
                for metric_name, metric_values in self.results['imp_quality'].items():
                    col_name = metric_name
                    df[col_name] = metric_values
                ret['imp_quality'] = df
                
            if 'local_pred' in self.results:
                df = pd.DataFrame()
                for metric_name, metric_values in self.results['local_pred'].items():
                    col_name = f'local_{metric_name}'
                    df[col_name] = metric_values
                ret['local_pred'] = df
                
            if 'fed_pred' in self.results:
                df = pd.DataFrame()
                for metric_name, metric_values in self.results['fed_pred']['personalized'].items():
                    col_name = f'personalized_{metric_name}'
                    df[col_name] = metric_values
                ret['fed_pred_personalized'] = df
                
                df = pd.DataFrame()
                num_clients = len(list(self.results['fed_pred']['personalized'].values())[0])
                for metric_name, metric_values in self.results['fed_pred']['global'].items():
                    col_name = f'global_{metric_name}'
                    df[col_name] = [metric_values]*num_clients    
                ret['fed_pred_global'] = df
            
            return ret
        
        ################################################################################################################
        # Export to one dictionary
        elif format == 'dict':
            return self.results

        ################################################################################################################
        # Text Table by Tabulate
        elif format == 'text-table':
            total_width = 63
            ret_str = ""
            ret_str += "=" * total_width + "\n"
            ret_str += "Evaluation Results" + "\n"
            ret_str += "=" * total_width + "\n"
            
            # Prepare data for tabulate
            headers = ["", "Average", "Std"]
            table_data = []

            # Add Imputation Quality metrics
            if 'imp_quality' in self.results:
                for metric, values in self.results['imp_quality']['imp_quality'].items():
                    mean = np.mean(values)
                    std = np.std(values)
                    if len(table_data) == 0:
                        table_data.append(["Imputation Quality", "", ""])
                    table_data.append([f"    {metric}", f"{mean:.3f}", f"{std:.3f}"])

            # Add horizontal separator
            if table_data:
                table_data.append(["-" * 29, "-" * 10, "-" * 10])

            # Add Downstream Prediction (Local) metrics
            if 'local_pred' in self.results:
                for metric, values in self.results['local_pred']['local_pred'].items():
                    mean = np.mean(values)
                    std = np.std(values)
                    if not any("Downstream Prediction (Local)" in row for row in table_data):
                        table_data.append(["Downstream Prediction (Local)", "", ""])
                    table_data.append([f"    {metric}", f"{mean:.3f}", f"{std:.3f}"])

            # Add horizontal separator
            if 'local_pred' in self.results:
                table_data.append(["-" * 29, "-" * 10, "-" * 10])

            # Add Downstream Prediction (Fed) metrics
            if 'fed_pred' in self.results:
                for metric, values in self.results['fed_pred']['global'].items():
                    mean = np.mean(values)
                    std = np.std(values)
                    if not any("Downstream Prediction (Fed)" in row for row in table_data):
                        table_data.append(["Downstream Prediction (Fed)", "", ""])
                    table_data.append([f"    {metric}", f"{mean:.3f}", "-"])

            # Print table using tabulate
            ret_str += tabulate(table_data, headers=headers, tablefmt="simple") + "\n"
            ret_str += "=" * total_width + "\n"
            return ret_str
        else:
            raise ValueError(f"Invalid format: {format}")
