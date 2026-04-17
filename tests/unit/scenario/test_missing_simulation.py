import numpy as np
import pytest

from fedimpute.scenario.missing_simulate.add_missing import add_missing, simulate_nan
from fedimpute.scenario.missing_simulate.add_missing_utils import (
    generate_missing_mech_funcs,
    generate_missing_ratios,
)
from fedimpute.scenario.missing_simulate.mar_simulate import simulate_nan_mar_quantile

pytestmark = pytest.mark.unit


def test_add_missing_mcar_preserves_shapes_and_adds_missing_values():
    clients_data = [
        np.linspace(0.0, 1.0, 40).reshape(10, 4),
        np.linspace(1.0, 2.0, 40).reshape(10, 4),
    ]
    rngs = [np.random.default_rng(1), np.random.default_rng(2)]

    clients_missing = add_missing(
        clients_data,
        cols=[0, 1, 2],
        rngs=rngs,
        obs_cols=[],
        global_missing=False,
        mf_strategy="all",
        mr_dist="random",
        mr_lower=0.4,
        mr_upper=0.4,
        mm_mech="mcar",
        seed=123,
    )
    
    # Check that the output has the same number of clients and columns, and that missing values were added
    assert [item.shape for item in clients_missing] == [(10, 3), (10, 3)]
    # Check that there are missing values in the output
    assert all(np.isnan(item).any() for item in clients_missing)


def test_add_missing_mcar_uses_client_level_missing_ratios():
    clients_data = [
        np.linspace(0.0, 1.0, 400).reshape(100, 4),
        np.linspace(1.0, 2.0, 400).reshape(100, 4),
    ]
    rngs = [np.random.default_rng(1), np.random.default_rng(2)]

    clients_missing = add_missing(
        clients_data,
        cols=[0, 1, 2],
        rngs=rngs,
        obs_cols=[],
        global_missing=False,
        mf_strategy="all",
        mr_dist="random",
        ms_mr_clients=[0.1, 0.3],
        mr_lower=0.0,
        mr_upper=1.0,
        mm_mech="mcar",
        seed=123,
    )

    assert np.isnan(clients_missing[0]).sum() == 30
    assert np.isnan(clients_missing[1]).sum() == 90


def test_mnar_sphere_uses_passed_rng_for_repeatable_masks():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=120)

    def run():
        return simulate_nan(
            X.copy(),
            y,
            mm_mech="mnar_logit",
            missing_features=[0, 1],
            missing_ratios=[0.3, 0.4],
            mechanism_funcs=["left", "right"],
            mm_strictness=True,
            mm_obs=False,
            mm_feature_option="all",
            mm_beta_option="sphere",
            rng=np.random.default_rng(999),
        )

    assert np.array_equal(run(), run(), equal_nan=True)


def test_mar_random_function_uses_passed_rng_for_repeatable_masks():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 4))

    def run():
        return simulate_nan_mar_quantile(
            X.copy(),
            cols=[0, 1],
            missing_ratio=[0.3, 0.4],
            missing_func="random",
            strict=True,
            rng=np.random.default_rng(999),
        )

    assert np.array_equal(run(), run(), equal_nan=True)


def test_missing_distribution_helpers_do_not_mutate_global_numpy_state():
    np.random.seed(321)
    expected = np.random.random(5)

    np.random.seed(321)
    generate_missing_ratios("random", [(0.2, 0.7)] * 3, 3, 4, seed=123)
    generate_missing_mech_funcs("random", "lr", 3, 4, seed=123)
    actual = np.random.random(5)

    assert np.allclose(actual, expected)
