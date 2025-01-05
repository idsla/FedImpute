import copy
import numpy as np


########################################################################################################################
# Local Initial Imputation
########################################################################################################################
def local(clients_data_utils, key='mean', col_type='num') -> List[np.ndarray]:
    if col_type == 'num':
        if key not in ['mean', 'median', 'zero']:
            raise ValueError("key must be one of 'mean', 'std', 'min', 'max'")

        if key == 'zero':
            clients_value_array = np.zeros((len(clients_data_utils), clients_data_utils[0]['num_cols']))
        else:
            clients_value_array = np.array(
                [[item['col_stats'][col][key] for col in range(item['num_cols'])] for item in clients_data_utils]
            )

    elif col_type == 'cat':
        if key not in ['mode', 'zero']:
            raise ValueError("key must be one of 'mode'")

        if key == 'zero':
            clients_value_array = np.zeros(
                (len(clients_data_utils), clients_data_utils[0]['n_features'] - clients_data_utils[0]['num_cols'])
            )
        else:
            clients_value_array = np.array(
                [[item['col_stats'][col][key] for col in range(item['num_cols'], item['n_features'])] for item in
                 clients_data_utils]
            )
    else:
        raise ValueError("col_type must be one of 'num', 'cat'")

    return [item for item in clients_value_array]


########################################################################################################################
# FedAvg Initial Imputation
########################################################################################################################
def fedavg(clients_data_utils, key='mean', col_type='num') -> List[np.ndarray]:

    if col_type == 'cat':
        if key not in ['mean']:
            raise ValueError("key must be one of 'mean', 'median'")

        sample_sizes = [
            [item['missing_stats_cols'][col]['sample_size_obs'] for col in range(item['num_cols'], item['n_features'])]
            for item in clients_data_utils
        ]  # (n_clients, )

        clients_value_array = np.array(
            [[item['col_stats'][col][key] for col in range(item['num_cols'], item['n_features'])]
             for item in clients_data_utils]
        )  # (n_clients, num_cols)

        # calculate weighted average
        imp_value_avg = np.average(
            clients_value_array, axis=0, weights=sample_sizes
        )

        # assign imputation values
        clients_impute_values = [imp_value_avg] * len(clients_data_utils)
        return clients_impute_values

    elif col_type == 'num':
        if key not in ['mean', 'median']:
            raise ValueError("key must be one of 'mean', 'median'")

        sample_sizes = [
            [item['missing_stats_cols'][col]['sample_size_obs'] for col in range(item['num_cols'])]
            for item in clients_data_utils
        ]  # (n_clients, )

        clients_value_array = np.array(
            [[item['col_stats'][col][key] for col in range(item['num_cols'])] for item in clients_data_utils]
        )  # (n_clients, num_cols)

        # calculate weighted average
        imp_value_avg = np.average(
            clients_value_array, axis=0, weights=sample_sizes
        )

        # assign imputation values
        clients_impute_values = [imp_value_avg] * len(clients_data_utils)
        return clients_impute_values


def central(clients_data_utils, key='mean', col_type='num') -> List[np.ndarray]:
    
    if col_type == 'num':
        if key not in ['mean', 'median', 'zero']:
            raise ValueError("key must be one of 'mean', 'std', 'min', 'max'")

        if key == 'zero':
            clients_value_array = np.zeros((len(clients_data_utils), clients_data_utils[0]['num_cols']))
        else:
            central_stats = clients_data_utils[-1]
            central_value = [central_stats['col_stats'][col][key] for col in range(central_stats['num_cols'])]
            clients_value_array = np.array(
                [copy.deepcopy(central_value) for item in clients_data_utils]
            )

    elif col_type == 'cat':
        if key not in ['mode', 'mean']:
            raise ValueError("key must be one of 'mode'")

        if key == 'zero':
            clients_value_array = np.zeros(
                (len(clients_data_utils), clients_data_utils[0]['n_features'] - clients_data_utils[0]['num_cols'])
            )
        else:
            central_stats = clients_data_utils[-1]
            central_value = [central_stats['col_stats'][col][key] for col in range(
                central_stats['num_cols'], central_stats['n_features'])]
            clients_value_array = np.array(
                [copy.deepcopy(central_value) for item in clients_data_utils]
            )
    else:
        raise ValueError("col_type must be one of 'num', 'cat'")

    return [item for item in clients_value_array]