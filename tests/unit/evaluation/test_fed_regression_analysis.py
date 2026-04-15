import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.regression.linear_model import OLSResults

from fedimpute.evaluation.fed_regression_analysis import (
    eval_fed_reg_linear,
    eval_fed_reg_logit,
)
from fedimpute.evaluation.fed_regression_analysis.linear import (
    distributed_qr_regression,
)
from fedimpute.evaluation.fed_regression_analysis.logit import (
    distributed_logistic_regression,
)

pytestmark = pytest.mark.unit


def _linear_client_data():
    X_1 = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 0.0, 1.0, 0.0]})
    X_2 = pd.DataFrame({"x1": [4.0, 5.0, 6.0, 7.0], "x2": [1.0, 0.0, 1.0, 0.0]})
    y_1 = pd.Series(1.0 + 2.0 * X_1["x1"] - 0.5 * X_1["x2"])
    y_2 = pd.Series(1.0 + 2.0 * X_2["x1"] - 0.5 * X_2["x2"])
    return [X_1, X_2], [y_1, y_2]


def _logit_client_data():
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 0])
    X_frame = pd.DataFrame(X, columns=["x1", "x2"])
    y_series = pd.Series(y)
    return [X_frame.iloc[:3].reset_index(drop=True), X_frame.iloc[3:].reset_index(drop=True)], [
        y_series.iloc[:3].reset_index(drop=True),
        y_series.iloc[3:].reset_index(drop=True),
    ]


def test_distributed_qr_regression_matches_exact_linear_coefficients():
    X_clients, y_clients = _linear_client_data()
    X_parts = [
        np.column_stack([np.ones(len(X_client)), X_client.values])
        for X_client in X_clients
    ]

    beta, cov_params = distributed_qr_regression(
        X_parts, [y_client.values for y_client in y_clients]
    )

    assert np.allclose(beta, np.array([1.0, 2.0, -0.5]))
    assert cov_params.shape == (3, 3)
    assert np.all(np.isfinite(cov_params))


def test_distributed_qr_regression_rejects_mismatched_partitions():
    X_clients, y_clients = _linear_client_data()

    with pytest.raises(ValueError, match="Number of X and y parts must match"):
        distributed_qr_regression([X_clients[0].values, X_clients[1].values], [y_clients[0].values])


def test_eval_fed_reg_linear_returns_statsmodels_result_with_expected_params():
    X_clients, y_clients = _linear_client_data()

    result = eval_fed_reg_linear(X_clients, y_clients)

    assert isinstance(result, OLSResults)
    assert result.model.exog_names == ["const", "x1", "x2"]
    assert np.allclose(result.params, np.array([1.0, 2.0, -0.5]))
    assert np.all(np.isfinite(result.bse))
    assert np.allclose(result.predict(result.model.exog), pd.concat(y_clients).values)


def test_distributed_logistic_regression_matches_centralized_logit_params():
    X_clients, y_clients = _logit_client_data()
    X = pd.concat(X_clients).values
    y = pd.concat(y_clients).values
    X_with_const = sm.add_constant(X)

    beta, hessian = distributed_logistic_regression(
        [X_with_const[:3], X_with_const[3:]],
        [y[:3], y[3:]],
        max_iter=100,
        tol=1e-8,
    )
    centralized = sm.Logit(y, X_with_const).fit(disp=0)

    assert np.allclose(beta, centralized.params)
    assert hessian.shape == (3, 3)
    assert np.all(np.isfinite(hessian))


def test_eval_fed_reg_logit_returns_binary_result_with_valid_predictions():
    X_clients, y_clients = _logit_client_data()

    result = eval_fed_reg_logit(X_clients, y_clients)
    predictions = result.predict()

    assert isinstance(result, BinaryResultsWrapper)
    assert result.model.exog_names == ["const", "x1", "x2"]
    assert np.all(np.isfinite(result.params))
    assert np.all(np.isfinite(result.bse))
    assert np.all((0 <= predictions) & (predictions <= 1))
