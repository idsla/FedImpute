from typing import List
import numpy as np
from .strategies import local, fedavg, central
from ...client.client import Client


def initial_imputation(strategy: str, Clients: List[Client]):

    if strategy == 'local':
        initial_imp_num = 'local_mean'
        initial_imp_cat = 'local_mode'
    elif strategy == 'fedavg':
        initial_imp_num = 'fedavg_mean'
        initial_imp_cat = 'fedavg_mean'
    elif strategy == 'central':
        initial_imp_num = 'central_mean'
        initial_imp_cat = 'central_mode'
    elif strategy == 'zero':
        initial_imp_num = 'zero'
        initial_imp_cat = 'zero'
    else:
        raise ValueError("strategy must be one of 'local', 'avg'")

    initial_data_num = initial_imputation_num(initial_imp_num, [client.data_utils for client in Clients])
    initial_data_cat = initial_imputation_cat(initial_imp_cat, [client.data_utils for client in Clients])

    for client_idx, client in enumerate(Clients):
        client.initial_impute(initial_data_num[client_idx], col_type='num')
        client.initial_impute(initial_data_cat[client_idx], col_type='cat')

    return Clients


########################################################################################################################
# Initial Imputation for Numerical Columns
def initial_imputation_num(strategy, clients_data_utils: List[dict]) -> List[np.ndarray]:
    if strategy == 'local_mean':
        return local(clients_data_utils, key='mean', col_type='num')
    elif strategy == 'local_median':
        return local(clients_data_utils, key='median', col_type='num')
    elif strategy == 'zero':
        return local(clients_data_utils, key='zero', col_type='num')
    elif strategy == 'fedavg_mean':
        return fedavg(clients_data_utils, key='mean')
    elif strategy == 'fedavg_median':
        return fedavg(clients_data_utils, key='median')
    elif strategy == 'central_mean':
        return central(clients_data_utils, key='mean', col_type='num')
    elif strategy == 'central_median':
        return central(clients_data_utils, key='median', col_type='num')
    elif strategy == 'complement_mean':
        raise NotImplemented
    elif strategy == 'complement_median':
        raise NotImplemented
    else:
        raise ValueError("strategy must be one of 'local_mean', 'local_median', 'local_zero', 'fedavg_mean', "
                         "'fedavg_median', 'complement_mean', 'complement_median'")


########################################################################################################################
# Initial Imputation for Categorical Columns
def initial_imputation_cat(strategy, clients_data_utils: List[dict]) -> List[np.ndarray]:
    if strategy == 'local_mode':
        return local(clients_data_utils, key='mode', col_type='cat')
    elif strategy == 'central_mode':
        return central(clients_data_utils, key='mode', col_type='cat')
    elif strategy == 'fedavg_mean':
        return fedavg(clients_data_utils, key='mean', col_type='cat')
    elif strategy == 'zero':
        return local(clients_data_utils, key='zero', col_type='cat')
    else:
        raise ValueError("strategy must be one of 'local_mode'")
