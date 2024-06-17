import numpy as np
import pytest

from src.imputation.imputers.simple_imputer import SimpleImputer
from collections import OrderedDict


class TestSimpleImputer:

    def test_simple_imputer(self):
        # initialization
        simple_imp = SimpleImputer(strategy='mean')
        data_utils = {'n_features': 3}
        simple_imp.initialize(data_utils, {}, 42)
        assert np.array_equal(simple_imp.mean_params, np.zeros(3))

        # test get_imp_model_params
        params = simple_imp.get_imp_model_params({})
        print(params)
        assert isinstance(params, dict) and isinstance(params['mean'], np.ndarray) and isinstance(params, OrderedDict)

        # test fit and impute
        X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=float)
        y = np.array([1, 2, 3])
        missing_mask = np.array([[False, False, False, False], [False, True, True, False], [True, False, False, True]])

        simple_imp.fit(X, y, missing_mask, {})
        print(simple_imp.mean_params)
        assert np.array_equal(simple_imp.mean_params, np.array([2.5, 5, 6, 5.5]))

        imputed = simple_imp.impute(X, y, missing_mask, {})
        print(imputed)
        assert np.array_equal(imputed, np.array([[1, 2, 3, 4], [4, 5, 6, 7], [2.5, 8, 9, 5.5]]))

        # test set_imp_model_params
        new_mean = np.array([1, 2, 3, 4])
        simple_imp.set_imp_model_params(OrderedDict({"mean": new_mean}), {})
        assert np.array_equal(simple_imp.mean_params, new_mean)

        imputed = simple_imp.impute(X, y, missing_mask, {})
        print(imputed)
        assert np.array_equal(imputed, np.array([[1, 2, 3, 4], [4, 2, 3, 7], [1, 8, 9, 4]]))
