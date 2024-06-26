from typing import List
import random
import pandas as pd

from .sampling import dirichlet_noniid_partition, generate_alphas
from sklearn.model_selection import train_test_split
import numpy as np


def data_partition(strategy, params, data, n_clients, seed=201030, regression=False) -> List[np.ndarray]:
    strategy, params = strategy.split('@')[0], dict([param.split('=') for param in strategy.split('@')[1:]])
    print(strategy, params)
    if strategy == 'full':
        return [data.copy() for _ in range(n_clients)]
    elif strategy == 'sample-evenly':
        sample_fracs = [1 / n_clients for _ in range(n_clients)]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret
    elif strategy == 'sample-unevenl1-1000':
        N = data.shape[0]
        sample_fracs = [0.5] + [1000/N for _ in range(n_clients - 1)]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        print([r.shape[0] for r in ret])
        return ret
    elif strategy == 'sample-unevenr1-1000':
        N = data.shape[0]
        sample_fracs = [1000/N for _ in range(n_clients - 1)] + [0.5]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        print([r.shape[0] for r in ret])
        return ret
    elif strategy == 'sample-unevenl1-600':
        N = data.shape[0]
        sample_fracs = [0.5] + [600/N for _ in range(n_clients - 1)]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        print([r.shape[0] for r in ret])
        return ret
    elif strategy == 'sample-unevenr1-600':
        N = data.shape[0]
        sample_fracs = [600/N for _ in range(n_clients - 1)] + [0.5]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        print([r.shape[0] for r in ret])
        return ret
    elif strategy == 'sample-unevendirl1':
        # get sample sizes
        sample_seed = 211
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        ratio = 0.2
        N = data.shape[0]
        n1_size = int(N * ratio)
        nrest_size = noniid_sample_dirichlet(N - n1_size, n_clients - 1, 0.1, 50, n1_size)
        sample_sizes = [n1_size] + nrest_size
        sample_fracs = [sample_size / N for sample_size in sample_sizes]

        # sample clients data
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret
    elif strategy == 'sample-unevendirr1':
        # get sample sizes
        sample_seed = 211
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        ratio = 0.2
        N = data.shape[0]
        n1_size = int(N * ratio)
        nrest_size = noniid_sample_dirichlet(N - n1_size, n_clients - 1, 0.1, 50, n1_size)
        sample_sizes = nrest_size + [n1_size]
        sample_fracs = [sample_size / N for sample_size in sample_sizes]

        # sample clients data
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret
    elif strategy == 'sample-uneven10':
        sample_seed = 211 + seed
        random.seed(sample_seed)
        # cluster 1
        n1 = random.randint(100, 500)
        n2 = random.randint(100, 500)
        n3 = random.randint(1000, 3000)
        n4 = random.randint(1000, 3000)
        n5 = data.shape[0]//2 - n1 - n2 - n3 - n4
        sample_fracs_sub1 = [n1, n2, n3, n4, n5]

        # cluster 2
        sample_seed = 212 + seed
        random.seed(sample_seed)
        n1 = random.randint(100, 500)
        n2 = random.randint(100, 500)
        n3 = random.randint(1000, 3000)
        n4 = random.randint(1000, 3000)
        n5 = data.shape[0]//2 - n1 - n2 - n3 - n4

        sample_fracs_sub2 = [n1, n2, n3, n4, n5]
        sample_fracs = sample_fracs_sub1 + sample_fracs_sub2
        sample_fracs = [sample_frac / data.shape[0] for sample_frac in sample_fracs]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret
    elif strategy == 'sample-uneven10dir':
        sample_seed = 211 + seed
        np.random.seed(sample_seed)
        sizes = noniid_sample_dirichlet(data.shape[0], 10, 0.2, 400, data.shape[0]*0.5, seed = sample_seed)
        sample_fracs = sizes
        sample_fracs = [sample_frac / data.shape[0] for sample_frac in sample_fracs]
        ret = []
        print(sample_seed)
        np.sum(sample_fracs)
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret
    elif strategy == 'sample-uneven10range':
        sample_seed = 211 + seed
        random.seed(sample_seed)
        sample_fracs = []
        for i in range(3):
            sample_fracs.append(random.randint(3000, 4000))
        for i in range(4):
            n1 = random.randint(500, 1000)
            sample_fracs.append(n1)
        for i in range(3):
            n1 = random.randint(1500, 2000)
            sample_fracs.append(n1)

        np.random.seed(sample_seed)
        np.random.shuffle(sample_fracs)
        print(sample_fracs)
        sample_fracs = [sample_frac / 18000 for sample_frac in sample_fracs]
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        return ret

    elif strategy == 'sample-unevenhs':
        # get sample sizes
        np.random.seed(seed)
        sample_fracs = [9000/18000] + [1000 /18000 for _ in range(n_clients - 1)]
        np.random.shuffle(sample_fracs)
        print(sample_fracs)
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            new_seed = seed + idx * seed + 990983
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32)
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        data[:, :-1], data[:, -1], test_size=sample_frac,
                        random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
        print([r.shape[0] for r in ret])
        return ret

    # TODO: add parameters - alpha, min_samples, max_samples, ratio to configuration files
    # TODO: unit test for each of scenarios
    else:
        raise ValueError('partition strategy not found')


def noniid_sample_dirichlet(num_population, n_clients, alpha, min_samples, max_samples, max_repeat_times=5e6, seed = None):
    """
    Perform non-iid sampling using dirichlet distribution non-iidness control by alpha,
    larger alpha, more uniform distributed, smaller alpha, more skewed distributed
    :param num_population: number of samples in the population
    :param n_clients: number of clients
    :param alpha: dirichlet distribution parameter
    :param min_samples: minimum number of samples in each client
    :param max_samples: maximum number of samples in each client
    :param max_repeat_times: maximum number of times to repeat the sampling
    :return: list of number of samples in each client
    """

    min_size = 0
    max_size = np.inf
    repeat_times = 0
    sizes = None

    while min_size < min_samples or max_size > max_samples:
        repeat_times += 1
        np.random.seed(seed + repeat_times)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        sizes = num_population * proportions
        min_size = min(sizes)
        max_size = max(sizes)
        if repeat_times > max_repeat_times:
            print('max repeat times reached')
            raise ValueError('max repeat times reached')

    print(repeat_times, sizes)
    return list(sizes)
