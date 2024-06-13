import loguru
from typing import List, Union, Tuple
import numpy as np
import gc

from .utils import (
    calculate_data_partition_stats, generate_local_test_data,
    generate_global_test_data, binning_target, binning_features, noniid_sample_dirichlet, generate_samples_iid
)


def load_data_partition(
        data: np.ndarray,
        data_config: dict,
        num_clients: int,
        seeds: List[int],
        partition_strategy: str,
        split_cols_option: int = 0,
        niid_alpha: float = 0.2,
        size_niid_alpha: float = 0.2,
        min_samples: int = 100,
        max_samples: int = 2000,
        sample_iid_direct: bool = False,
        even_sample_size: int = 1000,
        local_test_size: float = 0.1,
        global_test_size: float = 0.1,
        local_backup_size: float = 0.05,
        reg_bins: int = 50,
        global_seed: int = 201031,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, List[List[Tuple[int, int]]]]:

    """
    Load data partition
    :param data: data
    :param data_config: data configuration
    :param num_clients: number of clients
    :param seeds: seed for each client
    :param global_seed: random seed
    :param partition_strategy: partition strategy
    :param split_col_idx: niid split based on which column
    :param size_strategy: size strategy
    :param size_niid_alpha: size niid alpha
    :param min_samples: minimum samples
    :param max_samples: maximum samples
    :param niid_alpha: non-iid alpha dirichlet distribution parameter
    :param even_sample_size: even sample size
    :param sample_iid_direct: whether directly sample iid without based on target
    :param local_test_size: local test ratio
    :param global_test_size: global test ratio
    :param local_backup_size: local backup data ratio
    :param reg_bins: regression bins
    :return: List of training data, List of test data, global test data, statistics
    """

    #############################################################################################################
    # split a global test data
    train_data, test_data = generate_global_test_data(data, data_config, test_size=global_test_size, seed=global_seed)

    #############################################################################################################
    # partition data
    # iid partition
    np.random.seed(global_seed)
    if partition_strategy == 'iid-even':
        sample_fracs = [1 / num_clients for _ in range(num_clients)]
        regression = data_config['task_type'] == 'regression'
        datas = generate_samples_iid(
            train_data, sample_fracs, seeds, global_seed=global_seed, sample_iid_direct=sample_iid_direct,
            regression=regression, reg_bins=reg_bins
        )
    elif partition_strategy == 'iid-dir':
        if max_samples == -1:
            max_samples = data.shape[0]
        rng = np.random.default_rng(global_seed)
        sizes = noniid_sample_dirichlet(
            data.shape[0], num_clients, size_niid_alpha, min_samples, max_samples, rng=rng
        )
        sample_fracs = [size / data.shape[0] for size in sizes]
        regression = data_config['task_type'] == 'regression'
        datas = generate_samples_iid(
            train_data, sample_fracs, seeds, global_seed=global_seed, sample_iid_direct=sample_iid_direct,
            regression=regression, reg_bins=reg_bins
        )
    elif partition_strategy == 'iid-hs':
        sample_fracs = [0.5] + [0.05 for _ in range(num_clients - 1)]
        np.random.shuffle(sample_fracs)
        regression = data_config['task_type'] == 'regression'
        datas = generate_samples_iid(
            train_data, sample_fracs, seeds, global_seed=global_seed, sample_iid_direct=sample_iid_direct,
            regression=regression, reg_bins=reg_bins
        )
    elif partition_strategy == 'iid-random':
        sample_fracs = np.random.uniform(
            min_samples / data.shape[0], max_samples / data.shape[0], num_clients
        ).tolist()
        regression = data_config['task_type'] == 'regression'
        datas = generate_samples_iid(
            train_data, sample_fracs, seeds, global_seed=global_seed, sample_iid_direct=sample_iid_direct,
            regression=regression, reg_bins=reg_bins
        )

        # even partition with fixed sample size
        # elif size_strategy == 'even2':
        #     sample_fracs = [even_sample_size / data.shape[0] for _ in range(num_clients)]

    #############################################################################################################
    # non-iid partition
    elif partition_strategy == 'niid-dir':

        if isinstance(split_cols_option, str):
            if split_cols_option == 'target':
                split_col_idx = data.shape[1] - 1
            elif split_cols_option == 'first':
                split_col_idx = 0
            elif split_cols_option == 'random':
                np.random.seed(global_seed)
                split_col_idx = np.random.randint(0, data.shape[1] - 1)
            else:
                raise ValueError(f'Invalid split_col options: {split_cols_option}')
        elif isinstance(split_cols_option, list):
            if len(split_cols_option) > 1 and isinstance(split_cols_option[0], int):
                split_col_idx = split_cols_option[0]
            else:
                raise ValueError(f'Invalid split_cols, it either integer or list of integers or options ("target")')
        elif isinstance(split_cols_option, int):
            split_col_idx = split_cols_option
        else:
            raise ValueError(f'Invalid split_cols, it either integer or list of integers or options ("target")')
        # if split_col == 'target':
        #     split_col_idx = -1
        # elif split_col == 'feature':
        #     if 'split_col_idx' not in data_config:
        #         raise ValueError('split_col_idx is not provided')
        #     elif len(data_config['split_col_idx']) == 0:
        #         raise ValueError(
        #             'split_col_idx should have at least one split column index, when split col option is feature'
        #         )
        #     else:
        #         split_col_idx = data_config['split_col_idx'][0]
        # elif split_col == 'feature_cluster':
        #     if 'split_col_idx' not in data_config:
        #         raise ValueError('split_col_idx is not provided')
        #     elif len(data_config['split_col_idx']) == 0:
        #         raise ValueError(
        #             'split_col_idx should have at least one split column index, when split col option is feature'
        #         )
        #     else:
        #         split_col_idx = data_config['split_col_idx']
        # else:
        #     raise ValueError(
        #         'split_col_idx should have only one split column index, when split col option is feature'
        #     )

        datas = separate_data_niid(
            train_data, data_config, num_clients, split_col_idx, niid=True, partition='dir', balance=False,
            class_per_client=None, niid_alpha=niid_alpha, min_samples=min_samples, reg_bins=reg_bins,
            seed=global_seed
        )
    elif partition_strategy == 'niid-path':
        raise NotImplementedError
    else:
        raise ValueError('Strategy not found')

    del train_data
    gc.collect()

    #############################################################################################################
    # calculate statistics
    regression = data_config['task_type'] == 'regression'
    statistics = calculate_data_partition_stats(datas, regression=regression)

    #############################################################################################################
    # generate local test data
    train_datas, backup_datas, test_datas = generate_local_test_data(
        datas, seeds=seeds, local_test_size=local_test_size, local_backup_size=local_backup_size,
        regression=regression
    )

    return train_datas, backup_datas, test_datas, test_data, statistics


def separate_data_niid(
        data: np.ndarray, data_config: dict, num_clients: int, split_col_idx: Union[int, list] = -1,
        niid: bool = True, partition: str = 'dir', balance: bool = False,
        class_per_client: Union[None, int] = None, niid_alpha: float = 0.1,
        min_samples: int = 50, reg_bins: int = 20, seed: int = 201030
):
    rng = np.random.default_rng(seed)

    # split based on target
    if split_col_idx == -1:
        dataset_label = data[:, -1]
        if data_config['task_type'] == 'regression':  # if regression task, bin the target # TODO: refactor this
            dataset_label = binning_target(dataset_label, reg_bins, seed)
    # split based on feature
    else:
        # split based on one feature
        if not isinstance(split_col_idx, list):
            dataset_label = data[:, split_col_idx]
            if np.unique(dataset_label).shape[0] > reg_bins:
                dataset_label = binning_target(dataset_label, reg_bins, seed)
        # split based on multiple features (feature clustering)
        else:
            X = data[: split_col_idx]
            dataset_label = binning_features(X, reg_bins=10, seed=seed)

    dataset_content, target = data[:, :-1], data[:, -1]

    loguru.logger.debug(f"{dataset_label}")
    num_classes = len(np.unique(dataset_label))
    loguru.logger.debug(f"{len(np.unique(dataset_label))}")
    # guarantee that each client must have at least one batch of data for testing.
    min_samples = int(min(min_samples, int(len(dataset_label) / num_clients / 2)))  # ?
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = rng.integers(
                    int(max(num_per / 10, min_samples / num_classes)), int(num_per), num_selected_clients - 1
                ).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(
                        dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0
                    )
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        N = len(dataset_label)

        try_cnt = 1
        idx_clients = [[] for _ in range(num_clients)]
        # class_condition = False
        while (min_size < min_samples):
            # if try_cnt > 1:
            #     print(f'Client data size does not meet the minimum requirement {min_samples}. '
            #           f'Try allocating again for the {try_cnt}-th time.')

            idx_clients = [[] for _ in range(num_clients)]
            # all_class_condition = np.zeros(num_classes, dtype=bool)
            for class_id in range(num_classes):
                class_indices = np.where(dataset_label == class_id)[0]
                # split classes indices into num_clients parts
                rng.shuffle(class_indices)
                alphas = np.repeat(niid_alpha, num_clients)
                proportions = rng.dirichlet(alphas)
                proportions = np.array(
                    [p * (len(idx_client) < N / num_clients) for p, idx_client in zip(proportions, idx_clients)]
                )
                proportions = proportions / proportions.sum()

                # limited numbers
                # num_array = (proportions * len(class_indices)).astype(int)
                # all_class_condition[class_id] = (num_array == 1).any()
                # print(class_id, num_array, all_class_condition[class_id])

                # [100, 110, 113, 135, 235, ..., 100]
                proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
                splited_idx = np.split(class_indices, proportions)

                # filter out classes only with one sample
                splited_idx_new = []
                for idx in splited_idx:
                    if len(idx) == 1:
                        splited_idx_new.append(np.array([], dtype=int))
                    else:
                        splited_idx_new.append(idx)

                idx_clients = [idx_client + idx.tolist() for idx_client, idx in zip(idx_clients, splited_idx_new)]
                min_size = min([len(item) for item in idx_clients])

            try_cnt += 1
            # class_condition = ~(all_class_condition.any())

        for j in range(num_clients):
            dataidx_map[j] = idx_clients[j]
    else:
        raise NotImplementedError

    # assign data
    datas = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        idxs = dataidx_map[client]
        datas[client] = np.concatenate([dataset_content[idxs], target[idxs].reshape(-1, 1)], axis=1).copy()

    return datas
