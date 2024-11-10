import numpy as np
import pytest
from src.modules.data_paritition.partition_data_utils import calculate_data_partition_stats


class TestCalculateDataPartitionStatsTestSuite:
    def test_calculate_data_partition_stats(self):
        datas = [
            np.array([[1, 1, 2, 3, 4]]).reshape(-1, 1), np.array([[3, 3, 5, 5, 5]]).reshape(-1, 1),
            np.array([[6, 6, 2, 2, 2]]).reshape(-1, 1)
        ]

        stats = calculate_data_partition_stats(datas, regression=False)

        assert stats == [[(1, 2), (2, 1), (3, 1), (4, 1)], [(3, 2), (5, 3)], [(2, 3), (6, 2)]]

    def test_calculate_data_partition_stats_reg(self):
        datas = [
            3 + 1.5 * np.random.randn(50, 2), 10 + np.random.randn(50, 2),
            20 + 1.5 * np.random.randn(50, 2)
        ]

        stats = calculate_data_partition_stats(datas, regression=True, reg_bins=10)

        assert True


def test_binning_features():
    assert False
