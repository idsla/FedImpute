import numpy as np
import pandas as pd
from ..client import Client
from ..server import Server
from typing import List, Tuple
from typing import Union, List


def setup_clients(
        clients_data: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], clients_seeds: List[int], data_config: dict,
        imp_model_name: str, imp_model_params: dict, fed_strategy: str, fed_strategy_client_params: dict,
        client_config: dict
) -> List[Client]:

    clients = []
    for client_id, (client_data, client_seed) in enumerate(zip(clients_data, clients_seeds)):
        client = Client(
            client_id, train_data=client_data[0].values, test_data=client_data[1].values, X_train_ms=client_data[2].values,
            data_config=data_config, imp_model_name=imp_model_name, imp_model_params=imp_model_params,
            fed_strategy=fed_strategy, fed_strategy_params=fed_strategy_client_params, seed=client_seed,
            client_config=client_config, columns=client_data[0].columns.tolist()
        )
        clients.append(client)

    return clients


def setup_server(
        fed_strategy: str, fed_strategy_params: dict,
        imputer_name: str, imputer_params: dict,
        global_test: pd.DataFrame, data_config: dict, server_config: dict
) -> Server:

    server = Server(
        fed_strategy, fed_strategy_params, imputer_name, imputer_params, global_test.values, data_config, server_config, columns=global_test.columns.tolist()
    )
    return server


# def load_workflow(
#         workflow_name: str,
#         workflow_params: dict,
# ) -> Union[WorkflowICE, WorkflowICEGrad, WorkflowJM]:
#     """
#     Load the workflow based on the workflow name
#     """
#     if workflow_name == 'ice':
#         return WorkflowICE(**workflow_params)
#     elif workflow_name == 'icegrad':
#         return WorkflowICEGrad(workflow_params)
#     elif workflow_name == 'vae':
#         return WorkflowJM(workflow_params)
#     else:
#         raise ValueError(f"Workflow {workflow_name} not supported")
