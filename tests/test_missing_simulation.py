import numpy as np

from fedimpute.scenario.missing_simulate.add_missing import add_missing


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
        mr_dist="fixed",
        mr_lower=0.4,
        mr_upper=0.4,
        mm_mech="mcar",
        seed=123,
    )
    
    # Check that the output has the same number of clients and columns, and that missing values were added
    assert [item.shape for item in clients_missing] == [(10, 3), (10, 3)]
    # Check that there are missing values in the output
    assert all(np.isnan(item).any() for item in clients_missing)
