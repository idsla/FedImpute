import numpy as np
import pytest

from fedimpute.scenario.missing_simulate.add_missing_utils import (
    generate_missing_cols,
    generate_missing_mech,
    generate_missing_mech_funcs,
    generate_missing_ratios,
    resolve_ms_mr_clients,
)

pytestmark = pytest.mark.unit


def test_generate_missing_cols_all_repeats_columns_for_each_client():
    assert generate_missing_cols("all", num_clients=3, cols=[0, 2]) == [[0, 2], [0, 2], [0, 2]]


@pytest.mark.parametrize("strategy", ["random", "some", "none"])
def test_generate_missing_cols_rejects_unsupported_strategies(strategy):
    with pytest.raises(NotImplementedError):
        generate_missing_cols(strategy, num_clients=3, cols=[0, 1, 2])


def test_generate_missing_ratios_randu_equal_bounds_behaves_like_fixed_distribution():
    ratios = np.array(generate_missing_ratios("random", [(0.3, 0.3)] * 4, 4, 3, seed=123))

    assert ratios.shape == (4, 3)
    assert np.allclose(ratios, 0.3)


@pytest.mark.parametrize("dist", ["random", "normal", "random-int"])
def test_generate_missing_ratios_random_distributions_stay_in_range(dist):
    ratios = np.array(generate_missing_ratios(dist, [(0.2, 0.7)] * 5, 5, 4, seed=123))

    assert ratios.shape == (5, 4)
    assert np.nanmin(ratios) >= 0.2
    assert np.nanmax(ratios) <= 0.7


def test_resolve_ms_mr_clients_expands_scalar_for_each_client():
    assert resolve_ms_mr_clients(0.5, 3) == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]


def test_resolve_ms_mr_clients_expands_tuple_for_each_client():
    assert resolve_ms_mr_clients((0.2, 0.5), 2) == [(0.2, 0.5), (0.2, 0.5)]


def test_resolve_ms_mr_clients_accepts_float_list():
    assert resolve_ms_mr_clients([0.2, 0.3], 2) == [(0.2, 0.2), (0.3, 0.3)]


def test_resolve_ms_mr_clients_accepts_mixed_list():
    assert resolve_ms_mr_clients([0.5, (0.2, 0.3)], 2) == [(0.5, 0.5), (0.2, 0.3)]


def test_resolve_ms_mr_clients_accepts_bucket_list():
    assert resolve_ms_mr_clients(["extra-small", "large"], 2) == [(0.1, 0.2), (0.6, 0.8)]


@pytest.mark.parametrize(
    "ms_mr_clients",
    [
        [],
        [0.2],
        1.2,
        (0.5, 0.2),
        (0.1, 0.2, 0.3),
        ["unknown", "small"],
    ],
)
def test_resolve_ms_mr_clients_rejects_invalid_inputs(ms_mr_clients):
    with pytest.raises(ValueError):
        resolve_ms_mr_clients(ms_mr_clients, 2)


def test_generate_missing_ratios_respects_per_client_ranges():
    ranges = [(0.1, 0.2), (0.5, 0.6), (0.8, 0.9)]
    ratios = np.array(generate_missing_ratios("random", ranges, 3, 5, seed=123))

    assert ratios.shape == (3, 5)
    for client_idx, (lower, upper) in enumerate(ranges):
        assert np.nanmin(ratios[client_idx]) >= lower
        assert np.nanmax(ratios[client_idx]) <= upper


def test_generate_missing_ratios_clips_to_hard_bounds():
    ratios = np.array(
        generate_missing_ratios(
            "random", [(0.05, 0.05), (0.95, 0.95)], 2, 3, seed=123, mr_lower=0.1, mr_upper=0.9
        )
    )

    assert np.allclose(ratios[0], 0.1)
    assert np.allclose(ratios[1], 0.9)

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
