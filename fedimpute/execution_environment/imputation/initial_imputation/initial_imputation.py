from typing import List, Tuple, Union
import numpy as np
from .strategies import local, fedavg, central
from ...client.client import Client
from ...server.server import Server

def initial_imputation(Clients: List[Client], server: Server):
    strategy = server.fed_strategy.initial_impute
    if strategy == 'local':
        initial_imp_num = 'local_mean'
        initial_imp_cat = 'local_mode'
    elif strategy == 'fedavg':
        initial_imp_num = 'fedavg_mean'
        initial_imp_cat = 'local_mode'  # TODO: fedavg mode is not implemented, currently using local mode
    elif strategy == 'central':
        initial_imp_num = 'central_mean'
        initial_imp_cat = 'central_mode'
    elif strategy == 'zero':
        initial_imp_num = 'zero'
        initial_imp_cat = 'zero'
    else:
        raise ValueError("strategy must be one of 'local', 'avg'")

    initial_num_clients, initial_num_server = initial_imputation_num(
        initial_imp_num, [client.data_utils for client in Clients], server.data_utils
    )
    initial_cat_clients, initial_cat_server = initial_imputation_cat(
        initial_imp_cat, [client.data_utils for client in Clients], server.data_utils
    )

    for client_idx, client in enumerate(Clients):
        client.initial_impute(initial_num_clients[client_idx], col_type='num')
        client.initial_impute(initial_cat_clients[client_idx], col_type='cat')
    
    server.initial_impute(initial_num_server, col_type='num')
    server.initial_impute(initial_cat_server, col_type='cat')

    return Clients, server


########################################################################################################################
# Initial Imputation for Numerical Columns
def initial_imputation_num(
    strategy, 
    clients_data_utils: List[dict],
    server_data_utils: dict
) -> Tuple[List[np.ndarray], np.ndarray]:
    
    if len(clients_data_utils) == 0:
        raise ValueError("No clients data utils provided")
    
    if strategy == 'local_mean':
        clients_imp = local(clients_data_utils, key='mean', col_type='num')
        server_imp = local([server_data_utils], key='mean', col_type='num')[0]
    elif strategy == 'local_median':
        clients_imp = local(clients_data_utils, key='median', col_type='num')
        server_imp = local([server_data_utils], key='median', col_type='num')[0]
    elif strategy == 'zero':
        clients_imp = local(clients_data_utils, key='zero', col_type='num')
        server_imp = local([server_data_utils], key='zero', col_type='num')[0]
    elif strategy == 'fedavg_mean':
        clients_imp = fedavg(clients_data_utils, key='mean')
        server_imp = clients_imp[-1]
    elif strategy == 'fedavg_median':
        clients_imp = fedavg(clients_data_utils, key='median')
        server_imp = clients_imp[-1]
    elif strategy == 'central_mean':
        clients_imp = central(clients_data_utils, key='mean', col_type='num')
        server_imp = clients_imp[-1]
    elif strategy == 'central_median':
        clients_imp = central(clients_data_utils, key='median', col_type='num')
        server_imp = clients_imp[-1]
    elif strategy == 'complement_mean':
        raise NotImplemented
    elif strategy == 'complement_median':
        raise NotImplemented
    else:
        raise ValueError("strategy must be one of 'local_mean', 'local_median', 'local_zero', 'fedavg_mean', "
                         "'fedavg_median', 'complement_mean', 'complement_median'")
    
    return clients_imp, server_imp


########################################################################################################################
# Initial Imputation for Categorical Columns
def initial_imputation_cat(
    strategy, 
    clients_data_utils: List[dict],
    server_data_utils: dict
) -> Tuple[List[np.ndarray], np.ndarray]:
    
    if strategy == 'local_mode':
        clients_imp = local(clients_data_utils, key='mode', col_type='cat')
        server_imp = local([server_data_utils], key='mode', col_type='cat')[0]
    elif strategy == 'central_mode':
        clients_imp = central(clients_data_utils, key='mode', col_type='cat')
        server_imp = clients_imp[-1]
    elif strategy == 'zero':
        clients_imp = local(clients_data_utils, key='zero', col_type='cat')
        server_imp = local([server_data_utils], key='zero', col_type='cat')[0]
    else:
        raise ValueError("strategy must be one of 'local_mode', 'central_mode', 'zero'")
    
    return clients_imp, server_imp
