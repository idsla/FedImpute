from collections import OrderedDict
from copy import deepcopy

import loguru
import numpy as np
import multiprocessing as mp

from .utils import formulate_centralized_client, update_clip_threshold
from .workflow import BaseWorkflow
from fedimpute.execution_environment.server import Server
from typing import List
from fedimpute.execution_environment.client import Client
from fedimpute.execution_environment.utils.evaluator import Evaluator

from tqdm.auto import trange
from fedimpute.execution_environment.utils.tracker import Tracker
from ..imputation.initial_imputation.initial_imputation import initial_imputation
from .parallel import client_process_func, server_process_func


class WorkflowEM(BaseWorkflow):

    def __init__(
            self,
            max_iterations: int = 100,
            convergence_thres: float = 0.003,
            local_epoch: int = 1,
            evaluation_interval: int = 1,
            save_model_interval: int = 10
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_thres = convergence_thres
        self.local_epoch = local_epoch
        self.evaluation_interval = evaluation_interval
        self.save_model_interval = save_model_interval
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker,
    ) -> Tracker:

        """
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        evaluation_interval = self.evaluation_interval
        max_iterations = self.max_iterations
        save_model_interval = self.save_model_interval

        ############################################################################################################
        # Centralized Initialization
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation and evaluation
        clients = initial_imputation(server.fed_strategy.initial_impute, clients)
        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )
        # Update Global clip thresholds
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        ########################################################################################################
        # federated EM imputation
        fit_params_list = [{
            "local_epoch": self.local_epoch, "convergence_thres": self.convergence_thres
        } for _ in range(len(clients))
        ]

        # central and local training
        if server.fed_strategy.name == 'central' or server.fed_strategy.name == 'local':
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            local_models, clients_fit_res = [], []
            for client_id in trange(len(clients), desc='Clients', colour='green'):
                client = clients[client_id]
                fit_params = fit_params_list[client_id]
                fit_params['local_epoch'] = max_iterations * self.local_epoch
                fit_params['save_model_interval'] = save_model_interval
                fit_params.update(fit_instruction[client_id])
                model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            global_models, agg_res = server.fed_strategy.aggregate_parameters(
                local_model_parameters=local_models, fit_res=clients_fit_res, params={}
            )

            for global_model, client in zip(global_models, clients):
                client.update_local_imp_model(global_model, params={})
                client.local_imputation(params={})

        # Federated Training
        else:
            clients_converged_signs = [False for _ in range(len(clients))]
            clients_local_models_temp = [None for _ in range(len(clients))]

            for iteration in trange(max_iterations, desc='Iterations', leave=False, colour='blue'):

                #####################################################################################################
                # client local train imputation model
                local_models, clients_fit_res = [], []
                fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])

                for client in clients:
                    fit_params = fit_params_list[client.client_id]
                    fit_params.update(fit_instruction[client.client_id])

                    # if converged, don't need to fit again
                    if clients_converged_signs[client.client_id]:
                        fit_params.update({'fit_model': False})

                    model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                    local_models.append(model_parameter)
                    clients_fit_res.append(fit_res)

                if iteration == 5:
                    clients_local_models_temp = deepcopy(local_models)

                # check if all clients converged
                if iteration > 5:
                    clients_converged_signs = self.check_convergence(
                        old_parameters=clients_local_models_temp, new_parameters=local_models,
                        tolerance=self.convergence_thres
                    )
                    clients_local_models_temp = deepcopy(local_models)

                    # all converged
                    if all(clients_converged_signs):
                        loguru.logger.info(f"All clients converged, iteration {iteration}")
                        break

                #####################################################################################################
                # server aggregate local imputation models
                global_models, agg_res = server.fed_strategy.aggregate_parameters(
                    local_model_parameters=local_models, fit_res=clients_fit_res, params={}
                )

                for global_model, client in zip(global_models, clients):
                    if clients_converged_signs[client.client_id]:
                        continue
                    client.update_local_imp_model(global_model, params={})
                    client.local_imputation(params={})
                    if iteration % save_model_interval == 0:
                        client.save_imp_model(version=f'{iteration}')

                ########################################################################################################
                # Impute and Evaluation
                if iteration % evaluation_interval == 0:
                    self.eval_and_track(
                        evaluator, tracker, clients, phase='round', epoch=iteration, central_client=False
                    )

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(
            evaluator, tracker, clients, phase='final', central_client=server.fed_strategy.name == 'central'
        )
        for client in clients:
            client.save_imp_model(version='final')

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        ############################################################################################################
        # Workflow Parameters
        evaluation_interval = self.evaluation_interval
        max_iterations = self.max_iterations
        save_model_interval = self.save_model_interval

        ############################################################################################################
        # Centralized Initialization
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation and evaluation
        clients = initial_imputation(server.fed_strategy.initial_impute, clients)
        clients_data = [(client.X_train_imp, client.X_train, client.X_train_mask) for client in clients]
        self.eval_and_track_parallel(
            evaluator, tracker, clients_data, phase='initial', central_client=server.fed_strategy.name == 'central'
        )
        # Update Global clip thresholds
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        ########################################################################################################
        # federated EM imputation
        fit_params_list = [{
            "local_epoch": self.local_epoch, "convergence_thres": self.convergence_thres
        } for _ in range(len(clients))
        ]

        ############################################################################################################
        # Parallel Training for central and local
        if server.fed_strategy.name == 'central' or server.fed_strategy.name == 'local':

            client_pipes = [mp.Pipe() for _ in clients]
            server_pipe, main_pipe = mp.Pipe()
            client_processes = [mp.Process(
                target=client_process_func, args=(client, pipe[1])) for client, pipe in zip(clients, client_pipes)
            ]

            server_process = mp.Process(
                target=server_process_func, args=(server, [pipe[0] for pipe in client_pipes], server_pipe)
            )

            for p in client_processes + [server_process]:
                p.start()

            # client local training
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_id, (pipe, client) in enumerate(zip(client_pipes, clients)):
                fit_params = fit_params_list[client_id]
                fit_params['local_epoch'] = max_iterations * self.local_epoch
                fit_params['save_model_interval'] = save_model_interval
                fit_params.update(fit_instruction[client_id])
                pipe[0].send(("fit_local", fit_params))

            # Server aggregation
            main_pipe.send("aggregate")
            global_models, agg_res = main_pipe.recv()

            # Client update and local imputation
            for pipe, global_model in zip(client_pipes, global_models):
                pipe[0].send(("update_and_impute", {'global_model_params': global_model, 'params': {}}))

            # Receive client imputation results
            clients_data = [pipe[0].recv() for pipe in client_pipes]

            # Final Evaluation and Tracking and saving imputation model
            self.eval_and_track_parallel(
                evaluator, tracker, clients_data, phase='final', central_client=server.fed_strategy.name == 'central'
            )

            # Save Clients Model
            for pipe in client_pipes:
                pipe[0].send(("save_model", None))

            # Terminate processes and update clients and server
            for pipe, client in zip(client_pipes, clients):
                pipe[0].send(("terminate", None))
                new_client = pipe[0].recv()
                client.X_train_imp = new_client.X_train_imp
                client.X_train = new_client.X_train
                client.X_train_mask = new_client.X_train_mask
                client.imputer = new_client.imputer
                client.fed_strategy = new_client.fed_strategy
            main_pipe.send("terminate")
            new_server = main_pipe.recv()
            server.fed_strategy = new_server.fed_strategy

            # Join processes
            for p in client_processes + [server_process]:
                p.join()
                p.close()

            return tracker

        ################################################################################################################
        # Parallel Training for federated
        else:

            # setup client and server
            client_pipes = [mp.Pipe() for _ in clients]
            server_pipe, main_pipe = mp.Pipe()
            client_processes = [mp.Process(
                target=client_process_func, args=(client, pipe[1])) for client, pipe in zip(clients, client_pipes)
            ]

            server_process = mp.Process(
                target=server_process_func, args=(server, [pipe[0] for pipe in client_pipes], server_pipe)
            )

            for p in client_processes + [server_process]:
                p.start()

            # convergence signs
            clients_converged_signs = [False for _ in range(len(clients))]
            clients_local_models_temp = [None for _ in range(len(clients))]

            for iteration in trange(max_iterations, desc='Iterations', leave=False, colour='blue'):

                # client local training
                fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
                for client_id, (pipe, client) in enumerate(zip(client_pipes, clients)):
                    fit_params = fit_params_list[client_id]
                    fit_params.update(fit_instruction[client_id])
                    if clients_converged_signs[client.client_id]:
                        fit_params.update({'fit_model': False})
                    pipe[0].send(("fit_local", fit_params))

                # check convergence
                ret = [pipe[0].recv() for pipe in client_pipes]
                local_models = [item[0] for item in ret]
                fit_res_list = [item[1] for item in ret]
                if iteration == 5:
                    clients_local_models_temp = deepcopy(local_models)

                if iteration > 5:
                    clients_converged_signs = self.check_convergence(
                        old_parameters=clients_local_models_temp, new_parameters=local_models,
                        tolerance=self.convergence_thres
                    )
                    clients_local_models_temp = deepcopy(local_models)

                    # all converged
                    if all(clients_converged_signs):
                        loguru.logger.info(f"All clients converged, iteration {iteration}")
                        break

                main_pipe.send("aggregate")
                # send params and fit_res to server
                for pipe, params, fit_res in zip(client_pipes, local_models, fit_res_list):
                    pipe[1].send((params, fit_res))
                # Server aggregation
                global_models, agg_res = main_pipe.recv()

                # Client update and local imputation
                for client_id, (pipe, global_model) in enumerate(zip(client_pipes, global_models)):
                    if clients_converged_signs[client_id]:
                        continue
                    pipe[0].send(("update_and_impute", {'global_model_params': global_model, 'params': {}}))

                # Receive client imputation results and Evaluate
                clients_data = [pipe[0].recv() for pipe in client_pipes]
                if iteration % evaluation_interval == 0:
                    self.eval_and_track_parallel(
                        evaluator, tracker, clients_data, phase='round', epoch=iteration, central_client=False
                    )

                # Save model
                if iteration % save_model_interval == 0:
                    for pipe in client_pipes:
                        pipe[0].send(("save_model", None))

            ############################################################################################################
            # Terminate processes and update clients and server and collect environment
            for pipe, client in zip(client_pipes, clients):
                pipe[0].send(("terminate", None))
                new_client = pipe[0].recv()
                client.X_train_imp = new_client.X_train_imp
                client.X_train = new_client.X_train
                client.X_train_mask = new_client.X_train_mask
                client.imputer = new_client.imputer
                client.fed_strategy = new_client.fed_strategy
            main_pipe.send("terminate")
            new_server = main_pipe.recv()
            server.fed_strategy = new_server.fed_strategy

            # Join processes
            for p in client_processes + [server_process]:
                p.join()
                p.close()

            # Final Evaluation and Tracking and saving imputation model
            clients_data = [(client.X_train_imp, client.X_train, client.X_train_mask) for client in clients]
            self.eval_and_track_parallel(
                evaluator, tracker, clients_data, phase='final', central_client=server.fed_strategy.name == 'central'
            )

            return tracker

    @staticmethod
    def check_convergence(
            old_parameters: List[OrderedDict], new_parameters: List[OrderedDict], tolerance: float
    ) -> List[bool]:
        """
        Check convergence of the parameters
        """
        clients_converged = []
        for old_parameter, new_parameter in zip(old_parameters, new_parameters):
            mu, sigma = old_parameter['mu'], old_parameter['sigma']
            mu_new, sigma_new = new_parameter['mu'], new_parameter['sigma']
            converged = (
                    np.linalg.norm(mu - mu_new) < tolerance
                    and np.linalg.norm(sigma - sigma_new, ord=2) < tolerance
            )
            loguru.logger.debug(f"{np.linalg.norm(mu - mu_new)} {np.linalg.norm(sigma - sigma_new, ord=2)}")
            clients_converged.append(converged)

        return clients_converged
