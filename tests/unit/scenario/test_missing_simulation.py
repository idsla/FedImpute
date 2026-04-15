import numpy as np
import pytest

from fedimpute.scenario.missing_simulate.add_missing import add_missing

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
        mr_dist="randu",
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
        mr_dist="randu",
        ms_mr_clients=[0.1, 0.3],
        mr_lower=0.0,
        mr_upper=1.0,
        mm_mech="mcar",
        seed=123,
    )

    assert np.isnan(clients_missing[0]).sum() == 30
    assert np.isnan(clients_missing[1]).sum() == 90
