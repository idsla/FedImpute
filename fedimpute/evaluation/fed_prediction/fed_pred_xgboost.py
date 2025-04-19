
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from sklearn.linear_model import SGDClassifier, SGDRegressor
import loguru
import sys
import xgboost as xgb
from typing import Optional, Union, Tuple
import json
from tqdm import trange
from sklearn.metrics import f1_score, mean_squared_error
from copy import deepcopy
from sklearn.model_selection import train_test_split

from ...utils.nn_utils import EarlyStopping
from ...utils.reproduce_utils import set_seed
from ..pred_model_metrics import task_eval

def eval_fed_pred_xgboost(
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
    
    # Create DMatrix objects for each client's data, ensuring labels and features have matching dimensions
    train_dmatrices = []
    for X_train_imp, y_train in zip(X_train_imps, y_trains):
        train_dmatrices.append(xgb.DMatrix(X_train_imp, label=y_train.reshape(-1, 1)))
    test_dmatrix = xgb.DMatrix(X_test_global, label=y_test_global.reshape(-1, 1))
    val_dmatrix = xgb.DMatrix(X_global_val, label=y_global_val.reshape(-1, 1))
    
    ################################################################################################################
    # Model Parameters
    model_params_default = {
        'eta': 0.03,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'tree_method': 'hist',
        'random_state': seed,
    }
    
    model_params = {**model_params_default, **model_params}
    model_params['colsample_bytree'] = model_params['subsample']

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
        model_params['objective'] = 'binary:logistic' if clf_type == 'binary-class' else 'multi:softprob'
        if clf_type == 'multi-class':
            model_params['num_class'] = len(np.unique(y_test_global))
    else:
        eval_metrics = ['mse', 'mae', 'msle']
        model_params['objective'] = 'reg:squarederror'
        
    ################################################################################################################
    # Initialized Local and Global Models
    models = [
        xgb.train(
            params=model_params,
            dtrain=train_dmatrices[i],
            num_boost_round=local_epoch,
            evals=[(val_dmatrix, 'eval')],
            verbose_eval=False,
        ) for i in range(len(X_train_imps))
    ]
    model_bsts = [bytes(model.save_raw('json')) for model in models]
    
    global_bst = aggregate_trees(None, model_bsts)
    global_model = xgb.Booster(model_params).load_model(global_bst)
    models = [xgb.Booster(model_params) for _ in range(len(X_train_imps))]
    
    for model in models:
        model.load_model(global_bst)

    weights = [len(X_train_imp) for X_train_imp in X_train_imps]
    weights = [weight / sum(weights) for weight in weights]
    
     ################################################################################################################
    # Training    
    early_stopping_global = EarlyStopping(
        tolerance=tol, 
        tolerance_patience=patience, 
        increase_patience=patience,
        window_size=1, 
        check_steps=1, 
        backward_window_size=1
    )
    
    for epoch in trange(global_epoch, desc='Global Epoch', leave=False, colour='blue'):
        
        ############################################################################################################
        # Local training
        local_models = []
        for idx, (X_train_imp, y_train, clf_local) in enumerate(zip(X_train_imps, y_trains, models)):
            
            # fit local model
            local_bst = _local_boost(clf_local, train_dmatrices[idx], local_epoch)
            local_models.append(bytes(local_bst.save_raw('json')))

        ############################################################################################################
        # Server aggregation the local models
        global_bst = aggregate_trees(global_bst, local_models)
        global_model = xgb.Booster(model_params)
        global_model.load_model(global_bst)
        
        ############################################################################################################
        # Validate global model and update early stopping
        if task_type == 'classification':
            if clf_type == 'multi-class':
                y_pred_proba = global_model.predict(val_dmatrix)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred_proba = global_model.predict(val_dmatrix)
                y_pred = (y_pred_proba > 0.5).astype(float)
            val_loss = 1 - f1_score(y_global_val, y_pred.round(), average='macro')  # Convert to loss (lower is better)
        else:
            y_pred = global_model.predict(val_dmatrix)
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
            model.load_model(global_bst)

    ################################################################################################################
    # prediction and evaluation
    if task_type == 'classification':
        if clf_type == 'multi-class':
            y_pred_proba_global = global_model.predict(test_dmatrix)
            y_pred_global = np.argmax(y_pred_proba_global, axis=1)
        else:
            y_pred_proba_global = global_model.predict(test_dmatrix)
            y_pred_global = (y_pred_proba_global > 0.5).astype(float)
            y_pred_proba_global = np.array([1 - y_pred_proba_global, y_pred_proba_global]).T
    else:
        y_pred_global = global_model.predict(test_dmatrix)

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
    
################################################################################################################
# Utility Functions
################################################################################################################
def _local_boost(bst_input, train_dmatrix, num_local_round):
	
 	# Update trees based on local training data.
	for i in range(num_local_round):
		bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

	# Bagging: extract the last N=num_local_round trees for sever aggregation
	bst = (
		bst_input[
			bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()
		]
	)

	return bst

def aggregate_trees(
    global_model: Optional[bytes],
    model_params: List[bytes],
) -> bytearray:
    
    for model_param in model_params:
        global_model = aggregate(global_model, model_param)

    return bytearray(global_model)

def aggregate(
    bst_prev_org: Optional[bytes],
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    
    """Get the number of trees and parallel trees from the xgb model."""
    xgb_model = json.loads(bytearray(xgb_model_org))
    
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num