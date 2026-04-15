import numpy as np
import pandas as pd
import pytest

from fedimpute.data_prep.helper import one_hot_encoding, ordering_features
from fedimpute.utils.format_utils import arrays_to_dataframes, dataframe_to_numpy

pytestmark = pytest.mark.unit


def test_ordering_features_places_numerical_columns_before_target():
    data = pd.DataFrame(
        {
            "cat": ["a", "b"],
            "target": [0, 1],
            "num": [1.0, 2.0],
        }
    )

    ordered = ordering_features(data, numerical_cols=["num"], target_col="target")

    assert ordered.columns.tolist() == ["num", "cat", "target"]


def test_one_hot_encoding_keeps_target_as_last_column():
    data = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "target": [0, 1, 0],
        }
    )

    encoded = one_hot_encoding(data, numerical_cols_num=1)

    assert encoded.shape == (3, 3)
    assert np.array_equal(encoded[:, -1], np.array([0, 1, 0]))


def test_dataframe_to_numpy_factorizes_objects_when_target_is_last():
    data = pd.DataFrame(
        {
            "category": ["a", "b"],
            "value": [10.0, 20.0],
            "target": [1, 0],
        }
    )

    array, columns = dataframe_to_numpy(data, {"target": "target"})

    assert columns[-1] == "target"
    assert array.shape == (2, 3)
    assert set(array[:, 0]) == {0, 1}


def test_arrays_to_dataframes_can_drop_target_column():
    arrays = [np.array([[1.0, 2.0], [3.0, 4.0]])]

    frames = arrays_to_dataframes(arrays, columns=["x1", "x2", "y"], without_target=True)

    assert frames[0].columns.tolist() == ["x1", "x2"]
    assert frames[0].shape == (2, 2)
