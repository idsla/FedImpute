import loguru
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
from ..utils import nn_utils
from .parallel import client_process_func, server_process_func


class WorkflowICE(BaseWorkflow):

    def __init__(
            self,
            imp_iterations: int = 20,
            evaluation_interval: int = 1,
            early_stopping: bool = True,
            tolerance: float = 0.001,
            tolerance_patience: int = 3,
            increase_patience: int = 3,
            window_size: int = 3,
            log_interval: int = 1,
            save_model_interval: int = 5
    ):
        super(WorkflowICE, self).__init__()
        self.imp_iterations = imp_iterations
        self.evaluation_interval = evaluation_interval
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.tolerance_patience = tolerance_patience
        self.increase_patience = increase_patience
        self.window_size = window_size
        self.log_interval = log_interval
        self.save_model_interval = save_model_interval
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        """
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        data_dim = clients[0].X_train.shape[1]
        iterations = self.imp_iterations
        save_model_interval = self.save_model_interval

        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation and update clip threshold
        clients = initial_imputation(server.fed_strategy.initial_impute, clients)
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        # initial evaluation and tracking
        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        if server.fed_strategy.name == 'central':

            # centralized training
            early_stopping = nn_utils.EarlyStopping(
                tolerance_patience=self.tolerance_patience,
                increase_patience=self.increase_patience,
                tolerance=self.tolerance,
                window_size=self.window_size,
                check_steps=1,
                backward_window_size=1
            )

            central_client = clients[-1]
            for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):
                for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):
                    # client local train imputation model
                    fit_params = {'feature_idx': feature_idx, 'fit_model': True}
                    model_parameter, fit_res = central_client.fit_local_imp_model(params=fit_params)
                    central_client.update_local_imp_model(model_parameter, params={'feature_idx': feature_idx})
                    central_client.local_imputation(params={'feature_idx': feature_idx})

                    # broadcast model to other clients
                    for client in clients[:-1]:
                        client.update_local_imp_model(model_parameter, params={'feature_idx': feature_idx})
                        client.local_imputation(params={'feature_idx': feature_idx})

                # evaluation and early stopping and model saving
                imp_qualities = self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

                if epoch % save_model_interval == 0:
                    central_client.save_imp_model(version=f'{epoch}')

                if self.early_stopping:
                    central_imp_quality = imp_qualities[-1]
                    early_stopping.update(central_imp_quality)
                    if early_stopping.check_convergence():
                        loguru.logger.info(f"Central client converged, iteration {epoch}")
                        break

        else:
            ############################################################################################################
            # Federated Imputation Sequential Workflow
            all_clients_converged = [False for _ in range(len(clients))]
            early_stoppings = [
                nn_utils.EarlyStopping(
                    tolerance_patience=self.tolerance_patience,
                    increase_patience=self.increase_patience,
                    tolerance=self.tolerance,
                    window_size=self.window_size,
                    check_steps=1,
                    backward_window_size=1
                ) for _ in range(len(clients))
            ]

            fit_params_list = [{} for _ in range(len(clients))]

            for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):

                ########################################################################################################
                # federated imputation for each feature
                for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):
                    # client local train imputation model
                    local_models, clients_fit_res = [], []
                    fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
                    for client in clients:
                        fit_params = fit_params_list[client.client_id]
                        fit_params.update({'feature_idx': feature_idx})
                        fit_params.update(fit_instruction[client.client_id])
                        if all_clients_converged[client.client_id]:
                            fit_params.update({'fit_model': False})
                        model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                        local_models.append(model_parameter)
                        clients_fit_res.append(fit_res)

                    # server aggregate local imputation model
                    global_models, agg_res = server.fed_strategy.aggregate_parameters(
                        local_model_parameters=local_models, fit_res=clients_fit_res, params={}
                    )

                    # client update local imputation model and do imputation
                    for global_model, client in zip(global_models, clients):
                        if not all_clients_converged[client.client_id]:
                            client.update_local_imp_model(global_model, params={'feature_idx': feature_idx})
                            client.local_imputation(params={'feature_idx': feature_idx})

                ########################################################################################################
                # Evaluation and early stopping
                imp_qualities = self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

                if epoch % save_model_interval == 0:
                    for client in clients:
                        client.save_imp_model(version=f'{epoch}')

                if self.early_stopping:
                    for client_idx in range(len(clients)):
                        imp_quality = imp_qualities[client_idx]
                        early_stoppings[client_idx].update(imp_quality)
                        if early_stoppings[client_idx].check_convergence():
                            all_clients_converged[client_idx] = True
                            loguru.logger.debug(f"Client {client_idx} converged, iteration {epoch}")

                    if all(all_clients_converged):
                        loguru.logger.info(f"All clients converged, iteration {epoch}")
                        break

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

        """
        Imputation workflow for MICE Sequential Version
        """
        ################################################################################################################
        # Workflow Parameters
        data_dim = clients[0].X_train.shape[1]
        iterations = self.imp_iterations
        save_model_interval = self.save_model_interval

        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ################################################################################################################
        # Initial Imputation and update clip threshold
        clients = initial_imputation(server.fed_strategy.initial_impute, clients)
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        clients_data = [(client.X_train_imp, client.X_train, client.X_train_mask) for client in clients]
        self.eval_and_track_parallel(
            evaluator, tracker, clients_data, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ################################################################################################################
        # Setup Server and Client Process Function
        ################################################################################################################
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

        ################################################################################################################
        # Centralized Solution
        ################################################################################################################
        if server.fed_strategy.name == 'central':

            # centralized training
            early_stopping = nn_utils.EarlyStopping(
                tolerance_patience=self.tolerance_patience,
                increase_patience=self.increase_patience,
                tolerance=self.tolerance,
                window_size=self.window_size,
                check_steps=1,
                backward_window_size=1
            )

            pipe = client_pipes[-1]
            for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):
                for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):

                    # client local train imputation model
                    fit_params = {'feature_idx': feature_idx, 'fit_model': True}
                    pipe[0].send(("fit_local", fit_params))
                    model_parameter, fit_res = pipe[0].recv()
                    pipe[0].send(("update_and_impute", {
                        'global_model_params': model_parameter, 'params': {'feature_idx': feature_idx}
                    }))

                    # broadcast model to other clients
                    for client_pipe in client_pipes[:-1]:
                        client_pipe[0].send((
                            "update_and_impute",
                            {'global_model_params': model_parameter, 'params': {'feature_idx': feature_idx}}
                        ))

                    # evaluation and early stopping and model saving
                    clients_data = [pipe[0].recv() for pipe in client_pipes]

                # Save Clients Model
                if epoch % save_model_interval == 0:
                    for pipe in client_pipes:
                        pipe[0].send(("save_model", f'{epoch}'))

                # Final Evaluation and Tracking and saving imputation model
                imp_qualities = self.eval_and_track_parallel(
                    evaluator, tracker, clients_data, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

                # early stopping
                if self.early_stopping:
                    central_imp_quality = imp_qualities[-1]
                    early_stopping.update(central_imp_quality)
                    if early_stopping.check_convergence():
                        loguru.logger.info(f"Central client converged, iteration {epoch}")
                        break

        ################################################################################################################
        # Federated Solution
        ################################################################################################################
        else:
            all_clients_converged = [False for _ in range(len(clients))]
            early_stoppings = [
                nn_utils.EarlyStopping(
                    tolerance_patience=self.tolerance_patience,
                    increase_patience=self.increase_patience,
                    tolerance=self.tolerance,
                    window_size=self.window_size,
                    check_steps=1,
                    backward_window_size=1
                ) for _ in range(len(clients))
            ]

            fit_params_list = [{} for _ in range(len(clients))]
            for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):

                ########################################################################################################
                # federated imputation for each feature
                for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):

                    # client local train imputation model
                    fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
                    for client_idx, client in enumerate(clients):
                        fit_params = fit_params_list[client.client_id]
                        fit_params.update({'feature_idx': feature_idx})
                        fit_params.update(fit_instruction[client.client_id])
                        if all_clients_converged[client.client_id]:
                            fit_params.update({'fit_model': False})
                        pipe = client_pipes[client_idx]
                        pipe[0].send(("fit_local", fit_params))

                    # Server aggregation
                    main_pipe.send("aggregate")
                    global_models, agg_res = main_pipe.recv()

                    # client update local imputation model and do imputation
                    for client_idx, (global_model, client) in enumerate(zip(global_models, clients)):
                        if not all_clients_converged[client.client_id]:
                            pipe = client_pipes[client_idx]
                            pipe[0].send((
                                "update_and_impute",
                                {'global_model_params': global_model, 'params': {'feature_idx': feature_idx}}
                            ))
                        else:
                            pipe = client_pipes[client_idx]
                            pipe[0].send(("send_data", {'params': {'feature_idx': feature_idx}}))

                    clients_data = [pipe[0].recv() for pipe in client_pipes]

                ########################################################################################################
                # Evaluation and early stopping
                if epoch % save_model_interval == 0:
                    for pipe in client_pipes:
                        pipe[0].send(("save_model", f'{epoch}'))

                imp_qualities = self.eval_and_track_parallel(
                    evaluator, tracker, clients_data, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

                if self.early_stopping:
                    for client_idx in range(len(clients)):
                        imp_quality = imp_qualities[client_idx]
                        early_stoppings[client_idx].update(imp_quality)
                        if early_stoppings[client_idx].check_convergence():
                            all_clients_converged[client_idx] = True
                            loguru.logger.debug(f"Client {client_idx} converged, iteration {epoch}")

                    if all(all_clients_converged):
                        loguru.logger.info(f"All clients converged, iteration {epoch}")
                        break

        ########################################################################################################
        # Final Evaluation and Tracking
        for pipe in client_pipes:
            pipe[0].send(("save_model", 'final'))

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
