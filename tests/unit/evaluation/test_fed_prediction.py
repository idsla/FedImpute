import numpy as np
import pytest
import xgboost as xgb

from fedimpute.evaluation.fed_prediction import (
    eval_fed_pred_lr,
    eval_fed_pred_rf,
    eval_fed_pred_sklnn,
    eval_fed_pred_svm,
    eval_fed_pred_torchnn,
    eval_fed_pred_xgboost,
)
from fedimpute.evaluation.fed_prediction.fed_pred_xgboost import (
    _get_tree_nums,
    aggregate_trees,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings("ignore:Maximum number of iteration reached:sklearn.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning"),
]


def _binary_federated_prediction_data():
    X_train_imps = [
        np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [1.0, 1.0],
                [1.2, 0.9],
                [0.1, 0.2],
                [0.9, 1.1],
            ]
        ),
        np.array(
            [
                [0.0, 1.0],
                [0.1, 0.8],
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.9],
                [0.9, 0.1],
            ]
        ),
    ]
    y_trains = [
        np.array([0, 0, 1, 1, 0, 1]),
        np.array([0, 0, 1, 1, 0, 1]),
    ]
    X_tests = [
        np.array([[0.1, 0.0], [1.1, 1.0], [0.0, 0.2], [0.8, 1.0]]),
        np.array([[0.0, 0.9], [1.0, 0.1], [0.2, 0.8], [0.9, 0.0]]),
    ]
    y_tests = [
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
    ]

    return {
        "X_train_imps": X_train_imps,
        "y_trains": y_trains,
        "X_tests": X_tests,
        "y_tests": y_tests,
        "X_test_global": np.vstack(X_tests),
        "y_test_global": np.concatenate(y_tests),
        "data_config": {"task_type": "classification", "clf_type": "binary-class"},
    }


def _regression_federated_prediction_data():
    X_train_imps = [
        np.array(
            [
                [1.0, 1.0],
                [1.2, 1.1],
                [2.0, 2.0],
                [2.2, 2.1],
                [1.1, 1.2],
                [1.9, 2.1],
            ]
        ),
        np.array(
            [
                [1.0, 2.0],
                [1.1, 1.8],
                [2.0, 1.0],
                [1.8, 1.2],
                [1.2, 1.9],
                [1.9, 1.1],
            ]
        ),
    ]
    y_trains = [
        np.array([2.0, 2.3, 4.0, 4.3, 2.3, 4.0]),
        np.array([3.0, 2.9, 3.0, 3.0, 3.1, 3.0]),
    ]
    X_tests = [
        np.array([[1.1, 1.0], [2.1, 2.0], [1.0, 1.2], [1.8, 2.0]]),
        np.array([[1.0, 1.9], [2.0, 1.1], [1.2, 1.8], [1.9, 1.0]]),
    ]
    y_tests = [
        np.array([2.1, 4.1, 2.2, 3.8]),
        np.array([2.9, 3.1, 3.0, 2.9]),
    ]

    return {
        "X_train_imps": X_train_imps,
        "y_trains": y_trains,
        "X_tests": X_tests,
        "y_tests": y_tests,
        "X_test_global": np.vstack(X_tests),
        "y_test_global": np.concatenate(y_tests),
        "data_config": {"task_type": "regression", "clf_type": None},
    }


def _assert_binary_classification_fed_prediction_result(result, num_clients):
    assert set(result) == {"global", "personalized"}
    assert set(result["global"]) == {"accuracy", "f1", "auc", "prc"}
    assert set(result["personalized"]) == {"accuracy", "f1", "auc", "prc"}

    for values in result["global"].values():
        assert len(values) == 1
        assert np.isfinite(values[0])
        assert 0 <= values[0] <= 1

    for values in result["personalized"].values():
        assert len(values) == num_clients
        assert np.all(np.isfinite(values))
        assert np.all((0 <= np.array(values)) & (np.array(values) <= 1))


def _assert_regression_fed_prediction_result(result, num_clients):
    assert set(result) == {"global", "personalized"}
    assert set(result["global"]) == {"mse", "mae", "msle"}
    assert set(result["personalized"]) == {"mse", "mae", "msle"}

    for values in result["global"].values():
        assert len(values) == 1
        assert np.isfinite(values[0])
        assert values[0] >= 0

    for values in result["personalized"].values():
        assert len(values) == num_clients
        assert np.all(np.isfinite(values))
        assert np.all(np.array(values) >= 0)


@pytest.mark.parametrize(
    ("eval_func", "model_params", "train_params"),
    [
        (eval_fed_pred_lr, {}, {"global_epoch": 2, "val_ratio": 0.25}),
        (eval_fed_pred_svm, {"C": 1.0}, {"val_ratio": 0.25}),
        (
            eval_fed_pred_rf,
            {"n_estimators": 4, "max_depth": 2, "min_samples_leaf": 1},
            {"val_ratio": 0.25},
        ),
        (
            eval_fed_pred_sklnn,
            {"hidden_size": 4, "optimizer": "sgd", "lr": 0.01},
            {"global_epoch": 1, "local_epoch": 2, "val_ratio": 0.25},
        ),
        (
            eval_fed_pred_torchnn,
            {
                "batch_norm": False,
                "hidden_size": 4,
                "batch_size": 4,
                "optimizer": "sgd",
                "lr": 0.01,
            },
            {"global_epoch": 1, "local_epoch": 1, "val_ratio": 0.25},
        ),
        (
            eval_fed_pred_xgboost,
            {"max_depth": 1, "eta": 0.3},
            {"global_epoch": 1, "local_epoch": 1, "val_ratio": 0.25},
        ),
    ],
)
def test_fed_prediction_evaluators_return_fed_metric_contract(
    eval_func, model_params, train_params
):
    data = _binary_federated_prediction_data()

    result = eval_func(
        model_params=model_params,
        train_params=train_params,
        seed=1,
        verbose=0,
        **data,
    )

    _assert_binary_classification_fed_prediction_result(
        result, num_clients=len(data["X_train_imps"])
    )


@pytest.mark.parametrize(
    ("eval_func", "model_params", "train_params"),
    [
        (eval_fed_pred_lr, {"lr": 0.01}, {"global_epoch": 2, "val_ratio": 0.25}),
        (eval_fed_pred_svm, {"C": 1.0}, {"val_ratio": 0.25}),
        (
            eval_fed_pred_rf,
            {"n_estimators": 4, "max_depth": 2, "min_samples_leaf": 1},
            {"val_ratio": 0.25},
        ),
        (
            eval_fed_pred_sklnn,
            {"hidden_size": 4, "optimizer": "sgd", "lr": 0.001},
            {"global_epoch": 1, "local_epoch": 2, "val_ratio": 0.25},
        ),
        (
            eval_fed_pred_torchnn,
            {
                "batch_norm": False,
                "hidden_size": 4,
                "batch_size": 4,
                "optimizer": "sgd",
                "lr": 0.001,
            },
            {"global_epoch": 1, "local_epoch": 1, "val_ratio": 0.25},
        ),
        (
            eval_fed_pred_xgboost,
            {"max_depth": 1, "eta": 0.3},
            {"global_epoch": 1, "local_epoch": 1, "val_ratio": 0.25},
        ),
    ],
)
def test_fed_prediction_regression_evaluators_return_fed_metric_contract(
    eval_func, model_params, train_params
):
    data = _regression_federated_prediction_data()

    result = eval_func(
        model_params=model_params,
        train_params=train_params,
        seed=1,
        verbose=0,
        **data,
    )

    _assert_regression_fed_prediction_result(
        result, num_clients=len(data["X_train_imps"])
    )


def test_xgboost_tree_aggregation_combines_local_boosters():
    X_train = np.array(
        [[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.1, 0.9]],
        dtype=float,
    )
    y_train = np.array([0, 0, 1, 1])
    dmatrix = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": 1,
        "eta": 0.3,
        "seed": 1,
    }
    model_a = xgb.train(params=params, dtrain=dmatrix, num_boost_round=1)
    model_b = xgb.train(params=params, dtrain=dmatrix, num_boost_round=1)

    aggregated = aggregate_trees(
        None,
        [
            bytes(model_a.save_raw("json")),
            bytes(model_b.save_raw("json")),
        ],
    )

    tree_num, parallel_tree_num = _get_tree_nums(aggregated)
    assert tree_num == model_a.num_boosted_rounds() + model_b.num_boosted_rounds()
    assert parallel_tree_num == 1
