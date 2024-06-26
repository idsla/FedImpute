from src.modules.data_partition import data_partition
import numpy as np


def test_data_partition_uneven10dir():
    data = np.ones((18000, 10))
    for seed in range(10):
        ret = data_partition(
            strategy='sample-uneven10dir', params = None,  data=data, n_clients=10, seed=seed, regression=False
        )

        dims = []
        for item in ret:
            dims.append(item.shape[0])
        dims = np.array(dims)
        print(dims.max(), dims.min(), dims.mean(), np.quantile(dims, 0.5))



def test_data_partition_uneven10range():
    data = np.ones((18000, 10))

    ret = data_partition(
        strategy='sample-uneven10range', params = None,  data=data, n_clients=10, seed=102931466 + 10087, regression=False
    )

    for item in ret:
        print(item.shape)


def test_data_partition_unevenhs():
    data = np.ones((18000, 10))

    ret = data_partition(
        strategy='sample-unevenhs', params = None,  data=data, n_clients=10, seed=1203, regression=False
    )

    for item in ret:
        print(item.shape)
