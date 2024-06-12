import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, mean_squared_log_error
)


def task_eval(metric, task_type, clf_type, y_pred, y_test, y_pred_proba=None):
    if task_type == 'classification':
        if metric == 'accuracy':
            return np.mean(y_pred == y_test)
        elif metric == 'f1':
            if clf_type == 'binary-class' or clf_type == 'binary':
                return f1_score(y_test, y_pred)
            else:
                return f1_score(y_test, y_pred, average='weighted')
        elif metric == 'auc':
            assert y_pred_proba is not None, "y_pred_proba is None"
            if clf_type == 'binary-class' or clf_type == 'binary':
                return roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                return roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
        elif metric == 'prc':
            assert y_pred_proba is not None, "y_pred_proba is None"
            if clf_type == 'binary-class' or clf_type == 'binary':
                return average_precision_score(y_test, y_pred_proba[:, 1])
            else:
                return average_precision_score(y_test, y_pred_proba, average='weighted')
        else:
            raise ValueError(f"Invalid metric: {metric}")
    else:
        if metric == 'mse':
            return mean_squared_error(y_test, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_test, y_pred)
        elif metric == 'r2':
            return r2_score(y_test, y_pred)
        elif metric == 'msle':
            return mean_squared_log_error(y_test, y_pred)
        else:
            raise ValueError(f"Invalid metric: {metric}")