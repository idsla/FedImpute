
from typing import List, Dict, Tuple
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from sklearn.neural_network import MLPClassifier, MLPRegressor
import loguru
import sys
from tqdm import trange
from sklearn.metrics import f1_score, mean_squared_error
from copy import deepcopy
from sklearn.model_selection import train_test_split

from ...utils.nn_utils import EarlyStopping
from ...utils.reproduce_utils import set_seed
from ..pred_model_metrics import task_eval

def eval_fed_pred_sklnn(
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
        'val_ratio': 0.2,
        'log_interval': 10
	}
    
    train_params = {**train_params_default, **train_params}
    
    global_epoch = train_params['global_epoch']
    local_epoch = train_params['local_epoch']
    log_interval = train_params['log_interval']
    tol = train_params['tol']
    patience = train_params['patience']
    val_ratio = train_params['val_ratio']

    ################################################################################################################
    # Data Preparation
    # consruct validation set using 20% of X_train_imps and y_trains (merge) for early stopping
    X_train_merged = np.vstack(X_train_imps)
    y_train_merged = np.concatenate(y_trains)
    
    # Split merged training data into train and validation sets
    _, X_global_val, _, y_global_val = train_test_split(
        X_train_merged, y_train_merged, test_size=val_ratio, random_state=seed
    )
        
    ################################################################################################################
    # Model Parameters
    model_params_default = {
        'hidden_size': 32,
        'weight_decay': 0,
        'batch_size': 128,
        'optimizer': 'sgd',
        'lr': 0.001,
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
        global_model = MLPClassifier(
            hidden_layer_sizes=(model_params['hidden_size'], model_params['hidden_size']),
            alpha=model_params['weight_decay'],
            max_iter=local_epoch,
            warm_start=True,
            random_state=seed,
            learning_rate_init=model_params['lr'],
            solver=model_params['optimizer'],
        )
        
        n_classes = len(np.unique(y_test_global))
        n_features = X_test_global.shape[1]
        #global_model.classes_ = np.array(list(range(n_classes)))
        global_model.coef_ = np.zeros((1, n_features))
        global_model.intercept_ = np.zeros((1,))
    else:
        eval_metrics = ['mse', 'mae', 'msle']
        global_model = MLPRegressor(
            hidden_layer_sizes=(model_params['hidden_size'], model_params['hidden_size']),
            alpha=model_params['weight_decay'],
            learning_rate_init=model_params['lr'],
            max_iter=local_epoch,
            warm_start=True,
            random_state=seed,
            solver=model_params['optimizer'],
        )
        
        n_features = X_test_global.shape[1]
        global_model.coef_ = np.zeros((1, n_features))
        global_model.intercept_ = np.zeros((1,))

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
        # Local training
        losses = {}
        val_losses = {}
        for idx, (X_train_imp, y_train, clf_local) in enumerate(zip(X_train_imps, y_trains, models)):
            clf_local.fit(X_train_imp, y_train)

        ############################################################################################################
        # Server aggregation the local models
        coefs = []
        intercepts = []
        for idx, model in enumerate(models):
            coefs.append(model.coefs_)  
            intercepts.append(model.intercepts_)
        
        aggregated_coefs, aggregated_intercepts = [], []
        for i in range(len(coefs[0])):
            aggregated_coefs.append(np.average([coefs[j][i] for j in range(len(coefs))], axis=0, weights=weights))
            aggregated_intercepts.append(np.average([intercepts[j][i] for j in range(len(intercepts))], axis=0, weights=weights))
        
        ############################################################################################################
        # Validate global model and update early stopping
        global_model.fit(X_global_val, y_global_val)
        global_model.coefs_ = deepcopy(aggregated_coefs)
        global_model.intercepts_ = deepcopy(aggregated_intercepts)
        if task_type == 'classification':
            y_pred = global_model.predict(X_global_val)
            val_loss = 1 - f1_score(y_global_val, y_pred, average='macro')
        else:
            y_pred = global_model.predict(X_global_val)
            val_loss = mean_squared_error(y_global_val, y_pred)
        
        early_stopping_global.update(val_loss)
        
        if verbose >= 1:
            if epoch % log_interval == 0:
                loguru.logger.info(f"Epoch {epoch} - global val loss: {val_loss}")
        
        if early_stopping_global.check_convergence():
            if verbose >= 1:
                loguru.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        ############################################################################################################
        # local update
        for idx, model in enumerate(models):
            model.coef_ = aggregated_coefs
            model.intercept_ = aggregated_intercepts

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