import numpy as np
import pytest

from src.modules.missing_simulate.missing_scenario_utils import (
    generate_missing_cols, generate_missing_mech, generate_missing_mech_funcs,
    generate_missing_ratios
)


########################################################################################################################
class TestMissingCols:

    def test_generate_missing_cols_all(self):
        num_clients = 5
        cols = [0, 1, 2]
        expected_result = [cols] * num_clients
        result = generate_missing_cols('all', num_clients, cols)
        print(result)
        assert result == expected_result, f"Expected {expected_result} but got {result}"

    # Test for handling unsupported strategies
    @pytest.mark.parametrize("strategy", ['random', 'some', 'none'])
    def test_generate_missing_cols_unsupported(self, strategy):
        num_clients = 3
        cols = [0, 1, 2]
        with pytest.raises(NotImplementedError):
            generate_missing_cols(strategy, num_clients, cols)


#######################################################################################################################
@pytest.fixture
def setup():
    # Common setup shared across tests
    return {
        'num_clients': 10,
        'num_cols': 5,
        'seed': 201030
    }


class TestGenerateMissingRatios:
    def test_fixed_distribution(self, setup):
        missing_ratios = generate_missing_ratios(
            'fixed', (0.3, 0.3), setup['num_clients'], num_cols=setup['num_cols'], seed=setup['seed']
        )
        missing_ratios = np.array(missing_ratios)
        print('\n')
        assert np.array(missing_ratios).shape == (10, 5)
        assert np.all(missing_ratios == 0.3), "All values should be fixed at 0.3"

    def test_uniform_distribution(self, setup):
        missing_ratios = generate_missing_ratios(
            'uniform', (0.3, 0.7), setup['num_clients'], num_cols=setup['num_cols'], seed=setup['seed']
        )
        missing_ratios = np.array(missing_ratios)
        print('\n')
        assert np.array(missing_ratios).shape == (10, 5)
        assert missing_ratios.min() >= 0.3 and missing_ratios.max() <= 0.7, "Values should be within [0.3, 0.7]"

    def test_gaussian_distribution(self, setup):
        missing_ratios = generate_missing_ratios(
            'gaussian', (0.1, 0.7), setup['num_clients'], num_cols=setup['num_cols'], seed=setup['seed']
        )
        missing_ratios = np.array(missing_ratios)
        print('\n')
        assert np.array(missing_ratios).shape == (10, 5)
        assert missing_ratios.min() >= 0.1 and missing_ratios.max() <= 0.7, "Values should be within [0.1, 0.7]"

    def test_uniform_int_distribution(self, setup):
        missing_ratios = generate_missing_ratios(
            'uniform_int', (0.1, 0.7), setup['num_clients'], num_cols=setup['num_cols'], seed=setup['seed']
        )
        missing_ratios = np.array(missing_ratios)
        print(missing_ratios)
        print('\n')
        assert np.array(missing_ratios).shape == (10, 5)
        assert missing_ratios.min() >= 0.1 and missing_ratios.max() <= 0.7, "Values should be within [0.1, 0.7]"

    def test_gaussian_int_distribution(self, setup):
        missing_ratios = generate_missing_ratios(
            'gaussian_int', (0.2, 0.8), setup['num_clients'], num_cols=setup['num_cols'], seed=setup['seed']
        )
        missing_ratios = np.array(missing_ratios)
        print(missing_ratios)
        print('\n')
        assert missing_ratios.shape == (10, 5)
        assert missing_ratios.min() >= 0.2 and missing_ratios.max() <= 0.8, "Values should be within [0.2, 0.8]"


########################################################################################################################
class TestGenerateMissMech:

    @pytest.mark.parametrize("mech, expected_name", [
        ('mcar', 'mcar'),
        ('marq', 'mar_quantile'),
        ('marqst', 'mar_quantile_strict'),
        ('marsig', 'mar_sigmoid'),
        ('marsigst', 'mar_sigmoid_strict'),
        ('mnarq', 'mnar_quantile'),
        ('mnarqst', 'mnar_quantile_strict'),
        ('mnarsig', 'mnar_sigmoid'),
        ('mnarsigst', 'mnar_sigmoid_strict')
    ])
    def test_generate_missing_mech_valid(self, mech, expected_name):
        result = generate_missing_mech(mech, 10, 5, 123)
        assert (np.array(result) == expected_name).all(), f"Mechanism {mech} did not map correctly to {expected_name}"
        assert len(result) == 10 and all(len(row) == 5 for row in result), "Output dimensions are incorrect"

    def test_generate_missing_mech_invalid(self, ):
        with pytest.raises(ValueError):
            generate_missing_mech('invalid_mech', 10, 5, 123)


########################################################################################################################
# Test Missing Mech Distribution
########################################################################################################################
class TestMissingMechDistribution:
    def test_generate_missing_mech_dist_homo(self):
        result = generate_missing_mech_funcs("homo", 'lr', 10, 5, seed=123)
        result = np.array(result)
        print(result)
        assert all(len(np.unique(result[:, col_idx])) == 1 for col_idx in
                   range(result.shape[1])), "All rows should be identical in 'homo' distribution"

    def test_generate_missing_mech_dist_homo_none(self):
        result = generate_missing_mech_funcs("homo", None, 10, 5, seed=123)
        result = np.array(result)
        print(result)
        assert (result == None).all(), "All rows should be identical in 'homo' distribution"

    def test_generate_missing_mech_dist_random_none(self):
        with pytest.raises(ValueError):
            generate_missing_mech_funcs("random", None, 10, 5, seed=123)

    def test_generate_missing_mech_dist_random_singlefunc(self):
        with pytest.raises(ValueError):
            generate_missing_mech_funcs("random", 'l', 10, 5, seed=123)

    def test_generate_missing_mech_dist_random(self):
        result = generate_missing_mech_funcs("random", 'lr', 10, 5, seed=123)
        result = np.array(result)
        print(result)
        assert all(len(np.unique(result[:, col_idx])) == 2 for col_idx in range(result.shape[1])), 'value not correct'
        for i in range(result.shape[1]):
            for j in range(i + 1, result.shape[1]):
                assert not np.array_equal(result[:, i], result[:, j]), f"Columns {i} and {j} should not be identical"

    def test_generate_missing_mech_dist_random2(self):
        result = generate_missing_mech_funcs("random2", 'mt', 10, 5, seed=123)
        result = np.array(result)
        print(result)
        for i in range(result.shape[1]):
            for j in range(i + 1, result.shape[1]):
                assert not np.array_equal(result[:, i], result[:, j]), f"Columns {i} and {j} should not be identical"

    def test_generate_missing_mech_dist_invalid(self):
        with pytest.raises(ValueError):
            generate_missing_mech_funcs("invalid", None, 10, 5, seed=123)
