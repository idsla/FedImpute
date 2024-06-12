from typing import List, Tuple, Union
import math

import loguru
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans


def binning_target(y: np.ndarray, reg_bins: int, seed: int):
    """
    binning target variable
    :param y: target variable
    :param reg_bins: number of bins
    :param seed: random seed
    :return: binned target variable
    """
    assert reg_bins > 1, "reg_bins should be greater than 1"
    if np.unique(y).shape[0] <= reg_bins:
        return y
    y = y.copy().reshape(-1, 1)
    est = KBinsDiscretizer(
        n_bins=reg_bins, encode='ordinal', strategy='uniform',
        subsample=None, random_state=seed
    )
    y = est.fit_transform(y).flatten()
    return y


def binning_features(X, reg_bins=10, seed=0):
    """
    binning
    :param X: feature matrix
    :param reg_bins: number of bin
    :param seed: random seed
    :return: binned target variable
    """
    assert reg_bins > 1, "reg_bins should be greater than 1"
    X = X.copy().reshape(-1, 1)
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    est = KMeans(n_clusters=reg_bins, random_state=seed)
    clusters_label = est.fit_predict(X)
    return clusters_label


def calculate_data_partition_stats(
    datas: List[np.ndarray], regression: bool, reg_bins: int = 50
):
    """
    Calculate the statistics of the data partition
    :param datas: list of data partitions
    :param regression: regression task
    :param reg_bins: regression bins
    :return: list of statistics
    """
    if regression:
        label = np.concatenate([data[:, -1] for data in datas]).reshape(-1, 1)
        binner = KBinsDiscretizer(
            n_bins=reg_bins, encode='ordinal', strategy='uniform', subsample=None, random_state=0
        )
        binner.fit(label)

        num_clients = len(datas)
        statistic = [[] for _ in range(num_clients)]
        for client, data in enumerate(datas):
            target = binner.transform(data[:, -1].reshape(-1, 1)).flatten()
            for i in np.unique(target):
                statistic[client].append((int(i), int(sum((target == i).tolist()))))

        for client in range(num_clients):
            loguru.logger.debug(
                f"Client {client}\t Size of data: {len(datas[client])}\t Labels: ",
                np.unique(binner.transform(datas[client][:, -1].reshape(-1, 1)).flatten())
            )
            loguru.logger.debug(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            loguru.logger.debug("-" * 50)

    else:
        num_clients = len(datas)
        statistic = [[] for _ in range(num_clients)]
        for client, data in enumerate(datas):
            for i in np.unique(data[:, -1]):
                statistic[client].append((int(i), int(sum((data[:, -1] == i).tolist()))))

        for client in range(num_clients):
            loguru.logger.info(
                f"Client {client}\t Size of data: {len(datas[client])}\t Labels: ",
                np.unique(datas[client][:, -1])
            )
            loguru.logger.debug(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            loguru.logger.debug("-" * 50)

            # print(f"Client {client}\t Size of data: {len(datas[client])}\t Labels: ",
            #       np.unique(datas[client][:, -1]))
            # print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            # print("-" * 50)

    return statistic


def noniid_sample_dirichlet(
        num_population, n_clients, alpha, min_samples, max_samples, max_repeat_times=5e6,
        rng: np.random.Generator = np.random.default_rng(42)
):
    """
    Perform non-iid sampling using dirichlet distribution non-iidness control by alpha,
    larger alpha, more uniform distributed, smaller alpha, more skewed distributed
    :param rng: numpy random generator
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
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        sizes = num_population * proportions
        min_size = min(sizes)
        max_size = max(sizes)
        if repeat_times > max_repeat_times:
            loguru.logger.debug('max repeat times reached')
            raise ValueError('max repeat times reached')

    loguru.logger.debug(f"repeat time: {repeat_times} size: {sizes}")
    return list(sizes)


def generate_samples_iid(
        data: np.ndarray, sample_fracs: List[float], seeds: List[int], global_seed: int = 10023,
        sample_iid_direct: bool = False, regression: bool = False, reg_bins: Union[int, None] = None
) -> List[np.ndarray]:
    """
    generate samples iid
    :param data: global data array
    :param sample_fracs: sample size for each client
    :param seeds: randon seed for each client
    :param sample_iid_direct: whether directly iid sampling or not
    :param regression: whether regression task
    :param reg_bins: bins for a regression target
    :return: list of a data array for each client
    """
    if sample_iid_direct:  # directly sampling without iid based on target
        ret = []
        for idx, sample_frac in enumerate(sample_fracs):
            np.random.seed(seeds[idx])
            sampled_indices = np.random.choice(
                data.shape[0], size=math.ceil(data.shape[0] * sample_frac), replace=False
            )
            sampled_data = data[sampled_indices]
            ret.append(sampled_data)

        return ret
    else:  # iid based on target
        ret = []

        # set features and target
        X, y = data[:, :-1], data[:, -1]
        # if regression:
        #     y = binning_target(y, reg_bins, global_seed)

        # split using sample_fracs and train_test_split
        for idx, sample_frac in enumerate(sample_fracs):
            if sample_frac == 1.0:
                ret.append(data.copy())
            else:
                # new_seed = seed
                if regression:
                    _, X_test, _, y_test = train_test_split(
                        X, y, test_size=sample_frac, random_state=seeds[idx]
                    )
                else:
                    _, X_test, _, y_test = train_test_split(
                        X, y, test_size=sample_frac, random_state=seeds[idx], stratify=data[:, -1]
                    )
                ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1))

        return ret


def generate_local_test_data(
        datas: np.ndarray, seeds: List[int], local_test_size: float, local_backup_size: float, regression: bool
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.array]]:
    """
    Generate a local test data
    :param datas: list of data
    :param local_test_size: local test data ratio
    :param local_backup_size: retain a fraction of sample to ensure not all values are missing
    :param regression: regression task or not
    :param seeds: random seed for each client
    :return: tuple of train and test datas
    """

    train_datas, backup_datas, test_datas = [], [], []
    for idx, data in enumerate(datas):

        # split train and test
        if not regression:
            train_data, test_data = train_test_split(
                data, test_size=local_test_size, random_state=seeds[idx], stratify=data[:, -1]
            )
        else:
            train_data, test_data = train_test_split(
                data, test_size=local_test_size, random_state=seeds[idx]
            )

        # split train and retain
        if not regression:
            train_data, backup_data = train_test_split(
                data, test_size=local_backup_size, random_state=seeds[idx], stratify=data[:, -1]
            )
        else:
            train_data, backup_data = train_test_split(
                data, test_size=local_backup_size, random_state=seeds[idx]
            )

        train_datas.append(train_data)
        backup_datas.append(backup_data)
        test_datas.append(test_data)

    return train_datas, backup_datas, test_datas


def generate_global_test_data(
        data: np.ndarray, data_config: dict, test_size: float = 0.2, seed: int = 42):
    """
    Split data into train and test set
    """

    task_type = data_config['task_type']
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    train_data = np.concatenate([X_train, y_train], axis=1)
    test_data = np.concatenate([X_test, y_test], axis=1)

    return train_data, test_data
