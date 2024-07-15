import loguru
import multiprocessing as mp
from typing import List
from fedimpute.execution_environment.client import Client
from fedimpute.execution_environment.server import Server


def client_process_func(client: Client, client_pipe: mp.Pipe):
    """
    Client process for parallel federated imputation
    :param client: client object
    :param client_pipe: client pipe for communication
    :return: None
    """
    while True:
        command, data = client_pipe.recv()
        if command == "initial_impute":
            client.initial_impute(data)
        elif command == "fit_local":
            fit_params = data
            params, fit_res = client.fit_local_imp_model(fit_params)
            client_pipe.send((params, fit_res))
        elif command == "update_and_impute":
            client.update_local_imp_model(data['global_model_params'], params=data['params'])
            client.local_imputation(params=data['params'])
            client_pipe.send((client.X_train_imp, client.X_train, client.X_train_mask))
        elif command == "update_only":
            client.update_local_imp_model(data['global_model_params'], params=data['params'])
        elif command == "impute_only":
            client.local_imputation(params=data['params'])
            client_pipe.send((client.X_train_imp, client.X_train, client.X_train_mask))
        elif command == "save_model":
            version_name = data
            if version_name is not None:
                client.save_imp_model(version=version_name)
            else:
                client.save_imp_model(version='final')
        elif command == "terminate":
            client_pipe.send(client)
            break


def server_process_func(server: Server, client_pipes: List[mp.Pipe], server_pipe: mp.Pipe):
    """
    Server process for parallel federated imputation
    :param server: Server object
    :param client_pipes: client pipes for communication
    :param server_pipe: server pipe for communication
    :return: None
    """
    while True:
        command = server_pipe.recv()
        if command == "aggregate":
            params_list, fit_rest_list = [], []
            for pipe in client_pipes:
                params, fit_res = pipe.recv()
                params_list.append(params)
                fit_rest_list.append(fit_res)
            global_models, agg_res = server.fed_strategy.aggregate_parameters(params_list, fit_rest_list, {})
            server_pipe.send((global_models, agg_res))
        elif command == "terminate":
            server_pipe.send(server)
            break
