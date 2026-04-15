from collections import OrderedDict

import numpy as np
import pytest

from fedimpute.execution_environment.imputation.imputers.simple_imputer import SimpleImputer

pytestmark = pytest.mark.unit


def test_simple_imputer_fits_and_replaces_missing_values():
    imputer = SimpleImputer(strategy="mean")
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )
    y = np.array([0.0, 1.0, 0.0])
    missing_mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [True, False, False, True],
        ]
    )

    imputer.initialize(X.copy(), missing_mask, {"n_features": X.shape[1]}, {}, seed=42)
    assert np.array_equal(imputer.mean_params, np.zeros(X.shape[1]))

    fit_result = imputer.fit(X.copy(), y, missing_mask, {})
    assert fit_result["sample_size"] == 3
    assert np.allclose(imputer.mean_params, np.array([2.5, 5.0, 6.0, 5.5]))

    params = imputer.get_imp_model_params({})
    assert isinstance(params, OrderedDict)
    assert np.allclose(params["mean"], imputer.mean_params)

    imputed = imputer.impute(X.copy(), y, missing_mask, {})
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [2.5, 8.0, 9.0, 5.5],
        ]
    )
    assert np.allclose(imputed, expected)


def test_simple_imputer_accepts_global_model_parameters():
    imputer = SimpleImputer(strategy="mean")
    imputer.set_imp_model_params(OrderedDict({"mean": np.array([1.0, 2.0])}), {})

    X = np.array([[10.0, 20.0], [30.0, 40.0]])
    y = np.array([0.0, 1.0])
    missing_mask = np.array([[False, True], [True, False]])

    imputed = imputer.impute(X.copy(), y, missing_mask, {})
    assert np.allclose(imputed, np.array([[10.0, 2.0], [1.0, 40.0]]))
