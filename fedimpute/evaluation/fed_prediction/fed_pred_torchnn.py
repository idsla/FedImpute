
from typing import List, Dict, Tuple
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import loguru
import sys
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.metrics import f1_score, mean_squared_error
from copy import deepcopy

from ...utils.nn_utils import EarlyStopping
from ...utils.reproduce_utils import set_seed
from ..pred_model_metrics import task_eval
from ..twonn import TwoNNRegressor, TwoNNClassifier


def eval_fed_pred_torchnn(
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

    ################################################################################################################
    # Training Parameters
    train_params_default = {
        'global_epoch': 300,
        'local_epoch': 10,
        'tol': 0.001,
        'patience': 20,
        'batchnorm_avg': False,
        'val_ratio': 0.2,
        'log_interval': 10
	}
    
    train_params = {**train_params_default, **train_params}
    
    global_epoch = train_params['global_epoch']
    local_epoch = train_params['local_epoch']
    batchnorm_avg = train_params['batchnorm_avg']
    tol = train_params['tol']
    patience = train_params['patience']
    val_ratio = train_params['val_ratio']
    log_interval = train_params['log_interval']

    ################################################################################################################
    # Data Preparation
    # consruct validation set using 20% of X_train_imps and y_trains (merge) for early stopping
    X_train_merged = np.vstack(X_train_imps)
    y_train_merged = np.concatenate(y_trains)
    
    # Split merged training data into train and validation sets
    _, X_global_val, _, y_global_val = train_test_split(
        X_train_merged, y_train_merged, test_size=val_ratio, random_state=seed
    )
        
    model_params_default = {
        'optimizer': 'sgd',
        'weight_decay': 0,
        'batch_size': 128,
        'lr': 0.001,
        'hidden_size': 32,  
        'batch_norm': True,
        'dropout': 0.0
    }
    
    model_params = {**model_params_default, **model_params}

    try:
        task_type = data_config['task_type']
        clf_type = data_config['clf_type']
    except KeyError:
        raise KeyError("task_type is not defined in data_config")

    assert 'task_type' in data_config, "task_type is not defined in data_config"
    assert 'clf_type' in data_config, "clf_type is not defined in data_config"
    assert task_type in ['classification', 'regression'], f"Invalid task_type: {task_type}"
    
    task_type = data_config['task_type']
    clf_type = data_config['clf_type']
    
    ################################################################################################################
    # Loader classification model
    set_seed(seed)
    if task_type == 'classification':
        assert clf_type in ['binary-class', 'multi-class', 'binary'], f"Invalid clf_type: {clf_type}"
        eval_metrics = ['accuracy', 'f1', 'auc', 'prc']
        global_model = TwoNNClassifier(epochs=local_epoch, **model_params)
        unique_classes = np.unique(y_test_global)
        global_model._build_network(
            input_size=X_test_global.shape[1], 
            output_size=len(unique_classes), 
            class_weight=None, seed = seed
        )

    else:
        eval_metrics = ['mse', 'mae', 'msle']
        global_model = TwoNNRegressor(epochs=local_epoch, **model_params)
        global_model._build_network(
            input_size=X_test_global.shape[1], 
            seed = seed
        )


    ################################################################################################################
    # Evaluation
    models = [deepcopy(global_model) for _ in range(len(X_train_imps))]
    weights = [len(X_train_imp) for X_train_imp in X_train_imps]
    weights = [weight / sum(weights) for weight in weights]
    
    early_stopping_global = EarlyStopping(
        tolerance=tol, 
        tolerance_patience=patience, 
        increase_patience=patience,
        window_size=1, 
        check_steps=1, 
        backward_window_size=1
    )
    
    ################################################################################################################
    # Training
    for epoch in trange(global_epoch, desc='Global Epoch', leave=False, colour='blue'):
        
        ############################################################################################################
        # Local Training
        losses = {}
        val_losses = {}
        for idx, (X_train_imp, y_train, clf_local) in enumerate(zip(X_train_imps, y_trains, models)):
                    
            ret = clf_local.fit(X_train_imp, y_train, seed = seed)
            losses[idx] = ret['loss']
            #val_losses[idx] = ret['val_loss']

        # if verbose >= 1:
        #     if epoch % log_interval == 0:
        #         loguru.logger.info(
        #             f"Epoch {epoch} - train loss: {np.mean(list(losses.values()))} - val loss: {np.mean(list(val_losses.values()))}"
        #         )

        ############################################################################################################
        # Server Aggregation Local Models
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
        # Update global model parameters and early stopping
        global_model.update_parameters(aggregated_state_dict.copy())
        val_loss = global_model.validate(X_global_val, y_global_val, seed = seed)
        early_stopping_global.update(val_loss)
        
        if verbose >= 1:
            if epoch % log_interval == 0:
                loguru.logger.info(
                    f"Epoch {epoch} - global val loss: {val_loss}"
                )
        
        if early_stopping_global.check_convergence():
            if verbose >= 1:
                loguru.logger.info(f"Early stopping at epoch {epoch}")
            break
        
        ############################################################################################################
        # Local Update
        for idx, model in enumerate(models):
            model.update_parameters(aggregated_state_dict.copy())

    ################################################################################################################
    # prediction and evaluation
    y_pred_global = global_model.predict(X_test_global)
    if task_type == 'classification':
        y_pred_proba_global = global_model.predict_proba(X_test_global)
    else:
        y_pred_proba_global = None

    global_ret = {}
    for eval_metric in eval_metrics:
        if eval_metric not in global_ret:
            global_ret[eval_metric] = []
        global_ret[eval_metric].append(task_eval(
            eval_metric, task_type, clf_type, y_pred_global, y_test_global, y_pred_proba_global
        ))

    ret_personalized = {eval_metric: [] for eval_metric in eval_metrics}
    for X_train_imp, y_train, X_test, y_test in zip(X_train_imps, y_trains, X_tests, y_tests):
        y_pred = global_model.predict(X_test)
        if task_type == 'classification':
            y_pred_proba = global_model.predict_proba(X_test)
        else:
            y_pred_proba = None

        for eval_metric in eval_metrics:
            ret_personalized[eval_metric].append(task_eval(
                eval_metric, task_type, clf_type, y_pred, y_test, y_pred_proba
            ))

    return {
        'global': global_ret,
        'personalized': ret_personalized
    }