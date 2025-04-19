
from typing import List, Dict, Tuple
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from sklearn.svm import SVC, SVR
import loguru
import sys
from tqdm import trange
from sklearn.metrics import f1_score, mean_squared_error
from copy import deepcopy
from sklearn.model_selection import train_test_split

from ...utils.nn_utils import EarlyStopping
from ...utils.reproduce_utils import set_seed
from ..pred_model_metrics import task_eval

def eval_fed_pred_svm(
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
        'tol': 0.001,
        'patience': 20,
        'val_ratio': 0.2
	}
    
    train_params = {**train_params_default, **train_params}
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
    
    print(X_global_val.shape, y_global_val.shape)
    
    ################################################################################################################
    # Model Parameters
    model_params_default = {
        'kernel': 'rbf',
        'C': 1.0,
        'class_weight': 'balanced',
        'random_state': seed,
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
        global_model = SVC(
            kernel='rbf', C=model_params['C'], class_weight=model_params['class_weight'], 
            probability=True, random_state=seed
        )
        
        n_classes = len(np.unique(y_test_global))
        global_model.classes_ = np.array(list(range(n_classes)))
    else:
        eval_metrics = ['mse', 'mae', 'msle']
        global_model = SVR(
            kernel='rbf', C=model_params['C']
        )

    ################################################################################################################
    # Evaluation
    models = [deepcopy(global_model) for _ in range(len(X_train_imps))]
    weights = [len(X_train_imp) for X_train_imp in X_train_imps]
    weights = [weight / sum(weights) for weight in weights]
    
    ############################################################################################################
    # Local training
    val_losses = {}
    local_support_x = []
    local_support_y = []
    for idx, (X_train_imp, y_train, clf_local) in enumerate(zip(X_train_imps, y_trains, models)):
        
        # fit local model        
        clf_local.fit(X_train_imp, y_train)
        index = clf_local.support_
        local_support_x.append(X_train_imp[index])
        local_support_y.append(y_train[index])

    ############################################################################################################
    # Server aggregation Support Vectors and Fit Global Model
    global_support_x = np.concatenate(local_support_x, axis=0)
    global_support_y = np.concatenate(local_support_y, axis=0)
    global_model.fit(global_support_x, global_support_y)
    
    index = global_model.support_
    global_support_x = global_support_x[index]
    global_support_y = global_support_y[index]

    ############################################################################################################
    # Local refit with global support vectors
    for idx, model in enumerate(models):
        model.fit(global_support_x, global_support_y)

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