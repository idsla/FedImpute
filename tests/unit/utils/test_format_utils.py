import numpy as np
import pandas as pd
import pytest

from fedimpute.utils.format_utils import arrays_to_dataframes, dataframe_to_numpy

pytestmark = pytest.mark.unit


def test_dataframe_to_numpy_moves_target_to_last_and_factorizes_object_columns(capsys):
    data = pd.DataFrame(
        {
            "target": [1, 0, 1],
            "value": [10.0, 20.0, 30.0],
            "category": ["a", "b", "a"],
        }
    )

    array, columns = dataframe_to_numpy(data, {"target": "target"})

    assert columns == ["value", "category", "target"]
    assert np.array_equal(array[:, -1], np.array([1, 0, 1]))
    assert set(array[:, 1]) == {0, 1}
    assert "Object type column detected" in capsys.readouterr().out


def test_dataframe_to_numpy_rejects_missing_target_column():
    data = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    with pytest.raises(ValueError, match="list.remove"):
        dataframe_to_numpy(data, {"target": "target"})


def test_arrays_to_dataframes_preserve_columns_and_copy_input_arrays():
    arrays = [
        np.array([[1, 2, 0], [3, 4, 1]]),
        np.array([[5, 6, 1]]),
    ]

    frames = arrays_to_dataframes(arrays, columns=["x1", "x2", "target"])
    arrays[0][0, 0] = 999

    assert [frame.columns.tolist() for frame in frames] == [
        ["x1", "x2", "target"],
        ["x1", "x2", "target"],
    ]
    assert frames[0].iloc[0, 0] == 1.0
    assert all(dtype == float for dtype in frames[0].dtypes)


def test_arrays_to_dataframes_can_remove_target_column_from_each_frame():
    arrays = [
        np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]]),
        np.array([[5.0, 6.0, 1.0]]),
    ]

    frames = arrays_to_dataframes(arrays, columns=["x1", "x2", "target"], without_target=True)

    assert [frame.columns.tolist() for frame in frames] == [["x1", "x2"], ["x1", "x2"]]
    assert [frame.shape for frame in frames] == [(2, 2), (1, 2)]
