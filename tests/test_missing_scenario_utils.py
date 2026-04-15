import numpy as np
import pytest

from fedimpute.scenario.missing_simulate.add_missing_utils import (
    generate_missing_cols,
    generate_missing_mech,
    generate_missing_mech_funcs,
    generate_missing_ratios,
)


def test_generate_missing_cols_all_repeats_columns_for_each_client():
    assert generate_missing_cols("all", num_clients=3, cols=[0, 2]) == [[0, 2], [0, 2], [0, 2]]


@pytest.mark.parametrize("strategy", ["random", "some", "none"])
def test_generate_missing_cols_rejects_unsupported_strategies(strategy):
    with pytest.raises(NotImplementedError):
        generate_missing_cols(strategy, num_clients=3, cols=[0, 1, 2])


def test_generate_missing_ratios_fixed_distribution():
    ratios = np.array(generate_missing_ratios("fixed", (0.3, 0.3), 4, 3, seed=123))

    assert ratios.shape == (4, 3)
    assert np.allclose(ratios, 0.3)


@pytest.mark.parametrize("dist", ["randu", "randn", "randu-int", "randn-int"])
def test_generate_missing_ratios_random_distributions_stay_in_range(dist):
    ratios = np.array(generate_missing_ratios(dist, (0.2, 0.7), 5, 4, seed=123))

    assert ratios.shape == (5, 4)
    assert np.nanmin(ratios) >= 0.2
    assert np.nanmax(ratios) <= 0.7


@pytest.mark.parametrize(
    ("mech", "expected_name"),
    [
        ("mcar", "mcar"),
        ("marq", "mar_quantile"),
        ("marlogit", "mar_logit"),
        ("mnarq", "mnar_quantile"),
        ("mnarlogit", "mnar_logit"),
        ("mnarsmlogit", "mnar_sm_logit"),
    ],
)
def test_generate_missing_mech_maps_short_names(mech, expected_name):
    mechanisms = np.array(generate_missing_mech(mech, num_clients=2, num_cols=3, seed=123))

    assert mechanisms.shape == (2, 3)
    assert np.all(mechanisms == expected_name)


def test_generate_missing_mech_rejects_unknown_name():
    with pytest.raises(ValueError):
        generate_missing_mech("invalid", num_clients=2, num_cols=3, seed=123)


def test_generate_missing_mech_funcs_identity_uses_same_column_functions_across_clients():
    funcs = np.array(generate_missing_mech_funcs("identity", "lr", num_clients=4, num_cols=3, seed=123))

    assert funcs.shape == (4, 3)
    assert all(len(np.unique(funcs[:, col_idx])) == 1 for col_idx in range(funcs.shape[1]))


def test_generate_missing_mech_funcs_random_requires_multiple_function_options():
    with pytest.raises(ValueError):
        generate_missing_mech_funcs("random", "l", num_clients=4, num_cols=3, seed=123)
