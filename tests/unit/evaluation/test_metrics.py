import numpy as np
import pytest

from fedimpute.evaluation.imp_quality_metrics import mae, rmse
from fedimpute.evaluation.pred_model_metrics import task_eval

pytestmark = pytest.mark.unit


def test_imputation_quality_metrics_only_use_masked_values():
    X_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_imp = np.array([[1.0, 5.0], [2.0, 4.0]])
    mask = np.array([[False, True], [True, False]])

    assert mae(X_imp, X_true, mask) == 2.0
    assert rmse(X_imp, X_true, mask) == np.sqrt(5)


def test_imputation_quality_metrics_return_zero_without_missing_mask():
    X = np.array([[1.0, 2.0]])
    mask = np.array([[False, False]])

    assert mae(X, X, mask) == 0
    assert rmse(X, X, mask) == 0


def test_task_eval_classification_and_regression_metrics():
    y_test = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])

    assert task_eval("accuracy", "classification", "binary", y_pred, y_test) == 0.75
    assert task_eval("auc", "classification", "binary", y_pred, y_test, y_pred_proba) >= 0
    assert task_eval("mse", "regression", None, np.array([1.0, 3.0]), np.array([1.0, 1.0])) == 2.0


def test_task_eval_rejects_unknown_metric():
    with pytest.raises(ValueError):
        task_eval("bad", "classification", "binary", np.array([0]), np.array([0]))
