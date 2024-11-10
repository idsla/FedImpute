from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
)
from sklearn.linear_model import (
    BayesianRidge, LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV,
    LogisticRegression, Ridge, TheilSenRegressor, HuberRegressor, Lasso
)

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.base import BaseEstimator

from torch import nn

from ...utils.reproduce_utils import set_seed
from .models.ml_models.nn_model import (
    TwoNNRegressor, TwoNNClassifier
)
from .models.ml_models.linear_model import (
    RidgeRegression, Logistic
)


def load_sklearn_model(estimator_name) -> BaseEstimator:
    # TODO: ADD SEEDED RANDOM STATE
    if estimator_name == 'bayesian_ridge':
        return BayesianRidge()
    elif estimator_name == 'linear_regression':
        return LinearRegression(n_jobs=-1)
    elif estimator_name == 'ridge':
        return Ridge(alpha=1.0, random_state=0, solver='sag')
    elif estimator_name == 'bayesian_ridge':
        return BayesianRidge()
    elif estimator_name == 'lasso':
        return Lasso(alpha=0.1, random_state=0)
    elif estimator_name == 'theilsen':
        return TheilSenRegressor(random_state=0, n_jobs=-1)
    elif estimator_name == 'huber':
        return HuberRegressor()
    elif estimator_name == 'ridge_cv':
        return RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10, 50])
    elif estimator_name == 'lasso_cv':
        return LassoCV(alphas=[0.1, 1.0, 10.0])
    elif estimator_name == 'logistic':
        return LogisticRegression(penalty='l1', n_jobs=-1)
    elif estimator_name == 'logistic_cv':
        return LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], penalty='l1', solver='saga')
    elif estimator_name == 'mlp_reg':
        return MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)
    elif estimator_name == 'mlp_clf':
        return MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)
    elif estimator_name == 'dt_reg':
        return DecisionTreeRegressor(random_state=0)
    elif estimator_name == 'dt_clf':
        return DecisionTreeClassifier(random_state=0)
    elif estimator_name == 'rf_reg':
        return RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    elif estimator_name == 'rf_clf':
        return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    elif estimator_name == 'hist_reg':
        return HistGradientBoostingRegressor(random_state=0)
    elif estimator_name == 'hist_clf':
        return HistGradientBoostingClassifier(random_state=0)
    elif estimator_name == 'xgboost_reg':
        raise NotImplementedError
    elif estimator_name == 'xgboost_clf':
        raise NotImplementedError
    elif estimator_name == 'catboost_reg':
        raise NotImplementedError
    elif estimator_name == 'catboost_clf':
        raise NotImplementedError
    else:
        raise ValueError('Unknown estimator name: {}'.format(estimator_name))


def load_pytorch_model(model_name, model_params, seed) -> nn.Module:
    set_seed(seed)
    if model_name == 'nn_reg':
        return TwoNNRegressor(input_dim=model_params['input_dim'], hidden_dim=model_params['hidden_dim'])
    elif model_name == 'nn_clf':
        return TwoNNClassifier(input_dim=model_params['input_dim'], hidden_dim=model_params['hidden_dim'])
    elif model_name == 'ridge':
        return RidgeRegression(input_dim=model_params['input_dim'])
    elif model_name == 'logit':
        return Logistic(input_dim=model_params['input_dim'])
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))
