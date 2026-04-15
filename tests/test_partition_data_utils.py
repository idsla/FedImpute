import numpy as np

from fedimpute.scenario.data_partition.utils import (
    binning_target,
    calculate_data_partition_stats,
    generate_samples_iid,
)


def test_calculate_data_partition_stats_for_classification_labels():
    datas = [
        np.array([[0.0, 1.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0]]),
        np.array([[0.0, 3.0], [0.1, 3.0], [0.2, 5.0], [0.3, 5.0], [0.4, 5.0]]),
        np.array([[0.0, 6.0], [0.1, 6.0], [0.2, 2.0], [0.3, 2.0], [0.4, 2.0]]),
    ]

    stats = calculate_data_partition_stats(datas, regression=False)

    assert stats == [
        [(1, 2), (2, 1), (3, 1), (4, 1)],
        [(3, 2), (5, 3)],
        [(2, 3), (6, 2)],
    ]


def test_binning_target_preserves_small_number_of_unique_values():
    y = np.array([0.0, 1.0, 1.0, 0.0])

    binned = binning_target(y, reg_bins=10, seed=0)

    assert np.array_equal(binned, y)


def test_generate_samples_iid_returns_requested_client_partitions():
    X = np.arange(60, dtype=float).reshape(30, 2)
    y = np.array([0.0, 1.0] * 15).reshape(-1, 1)
    data = np.concatenate([X, y], axis=1)

    samples = generate_samples_iid(
        data,
        sample_fracs=[0.2, 0.3],
        seeds=[1, 2],
        global_seed=0,
        sample_iid_direct=False,
        regression=False,
    )

    assert [sample.shape for sample in samples] == [(6, 3), (9, 3)]
    assert all(set(np.unique(sample[:, -1])) == {0.0, 1.0} for sample in samples)
