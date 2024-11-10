import numpy as np

from src.loaders.load_missing_simulation_cent import add_missing_central


def run_test_add_missing_central(seed):

    np.random.seed(seed)  # Fix the seed for reproducibility
    data = np.random.rand(20, 10)  # Example data with 5 rows and 3 columns
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Columns where missing data should be simulated

    # Run the function under test
    result = add_missing_central(
        data, cols, 'mnarsig', 'lr',
        'uniform_int@mrl=0.3-mrr=0.6', 'all', seed
    )

    print(result)

    # Assert no errors and check structure
    assert result is not None, "Function should return a non-None result"
    assert result.shape == data.shape, "Output data should have the same shape as input data"

    # Check that missing values were added correctly
    assert np.any(np.isnan(result[:, 1])), "Column 2 should have NaN values"
    assert np.any(np.isnan(result[:, 2])), "Column 3 should have NaN values"


# Test function
def test_add_missing_central():
    run_test_add_missing_central(42)  # Using 42 as the fixed seed for consistency