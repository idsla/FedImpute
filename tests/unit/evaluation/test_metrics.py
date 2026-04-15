import numpy as np
import pytest

from fedimpute.evaluation.imp_quality_metrics import mae, rmse, sliced_ws, ws_cols
from fedimpute.evaluation.pred_model_metrics import task_eval

pytestmark = pytest.mark.unit


def _assert_finite_number(value):
    assert isinstance(value, (int, float, np.integer, np.floating))
    assert np.isfinite(value)


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


def test_distributional_imputation_quality_metrics_are_finite_numbers():
    X_true = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    X_imp = X_true + np.array(
        [
            [0.1, -0.1, 0.0],
            [0.0, 0.2, -0.1],
            [-0.2, 0.0, 0.1],
            [0.1, -0.1, 0.2],
        ]
    )

    for value in [ws_cols(X_imp, X_true), sliced_ws(X_imp, X_true, N=5, seed=123)]:
        _assert_finite_number(value)
        assert value >= 0


@pytest.mark.parametrize("metric", ["accuracy", "f1", "auc", "prc"])
def test_task_eval_binary_classification_metrics_used_by_evaluation_models(metric):
    y_test = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_pred_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.4, 0.6],
        ]
    )

    value = task_eval(metric, "classification", "binary", y_pred, y_test, y_pred_proba)

    _assert_finite_number(value)
    assert 0 <= value <= 1


@pytest.mark.parametrize("metric", ["accuracy", "f1", "auc", "prc"])
def test_task_eval_multiclass_classification_metrics_used_by_evaluation_models(metric):
    y_test = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_pred_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.2, 0.6],
        ]
    )

    value = task_eval(metric, "classification", "multi-class", y_pred, y_test, y_pred_proba)

    _assert_finite_number(value)
    assert 0 <= value <= 1


@pytest.mark.parametrize("metric", ["mse", "mae", "msle"])
def test_task_eval_regression_metrics_used_by_evaluation_models(metric):
    y_test = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.5, 2.0, 5.0, 7.0])

    value = task_eval(metric, "regression", None, y_pred, y_test)

    _assert_finite_number(value)
    assert value >= 0


def test_task_eval_regression_r2_metric_supported_by_task_eval():
    value = task_eval(
        "r2",
        "regression",
        None,
        y_pred=np.array([1.5, 2.0, 5.0, 7.0]),
        y_test=np.array([1.0, 2.0, 4.0, 8.0]),
    )

    _assert_finite_number(value)


def test_task_eval_rejects_unknown_metric():
    with pytest.raises(ValueError):
        task_eval("bad", "classification", "binary", np.array([0]), np.array([0]))
