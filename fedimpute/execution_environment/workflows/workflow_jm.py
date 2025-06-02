import logging
import loguru
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import fedimpute.execution_environment.utils.nn_utils as nn_utils

from fedimpute.execution_environment.server import Server
from typing import List
from fedimpute.execution_environment.client import Client
from fedimpute.execution_environment.utils.evaluator import Evaluator
from .utils import formulate_centralized_client, update_clip_threshold
from .workflow import BaseWorkflow
from tqdm.auto import trange

from ..imputation.initial_imputation.initial_imputation import initial_imputation
from ..utils.tracker import Tracker
from .parallel import client_process_func, server_process_func


class WorkflowJM(BaseWorkflow):

    def __init__(
            self,
            initial_zero_impute: bool = True,
            global_epoch: int = 300,
            local_epoch: int = 5,
            use_early_stopping: bool = True,
            log_interval: int = 10,
            imp_interval: int = 1000,
            save_model_interval: int = 200,
            model_converge: dict = {
                "tolerance": 0.001,
                "tolerance_patience": 20,
                "increase_patience": 20,
                "window_size": 20,
                "check_steps": 1,
                "back_steps": 1
            },
    ):
        super().__init__('JM (Joint Modeling)')
        self.tracker = None
        self.initial_zero_impute = initial_zero_impute
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.use_early_stopping = use_early_stopping
        self.model_converge = model_converge
        self.log_interval = log_interval
        self.imp_interval = imp_interval
        self.save_model_interval = save_model_interval

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        ############################################################################################################
        # Initial Imputation and Evaluation
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        # clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        if self.initial_zero_impute:
            server.fed_strategy.initial_impute = 'zero'
            clients, server = initial_imputation(clients, server)
        else:
            clients, server = initial_imputation(clients, server)
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ############################################################################################################
        # Federated Imputation Workflow
        use_early_stopping = self.use_early_stopping
        early_stopping_mode = 'local'

        model_converge_tol = self.model_converge['tolerance']
        model_converge_tolerance_patience = self.model_converge['tolerance_patience']
        model_converge_increase_patience = self.model_converge['increase_patience']
        model_converge_window_size = self.model_converge['window_size']
        model_converge_steps = self.model_converge['check_steps']
        model_converge_back_steps = self.model_converge['back_steps']

        early_stoppings, all_clients_converged_sign = self.setup_early_stopping(
            early_stopping_mode, model_converge_tol, model_converge_tolerance_patience,
            model_converge_increase_patience, model_converge_window_size, model_converge_steps,
            model_converge_back_steps, clients
        )

        ################################################################################################################
        # Federated Training
        global_model_epochs = self.global_epoch
        log_interval = self.log_interval
        imp_interval = self.imp_interval
        save_model_interval = self.save_model_interval

        fit_params_list = [{
            'local_epoch': self.local_epoch
        } for _ in clients]

        for epoch in trange(global_model_epochs, desc='Global Epoch', colour='blue'):

            ###########################################################################################
            # Local training of an imputation model
            local_models, clients_fit_res = [], []

            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                client = clients[client_idx]
                fit_params = fit_params_list[client_idx]
                fit_params.update(fit_instruction[client_idx])
                # if it is converged, do not fit the model
                if early_stopping_mode == 'local' and all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            ############################################################################################
            # Aggregate local imputation model
            global_models, agg_res = server.fed_strategy.aggregate_parameters(
                local_model_parameters=local_models, fit_res=clients_fit_res, params={
                    'current_epoch': epoch, 'global_epoch': global_model_epochs
                }
            )

            ###########################################################################################
            # Updates local imputation model and do imputation
            for client_idx, (global_model, client) in enumerate(zip(global_models, clients)):
                if early_stopping_mode == 'local' and (all_clients_converged_sign[client_idx]):
                    continue
                client.update_local_imp_model(global_model, params={})
                if epoch % save_model_interval == 0:
                    client.save_imp_model(version=f'{epoch}')
            
            # Server local imputation
            server.local_imputation(params={})

            #############################################################################################
            # Early Stopping, Loss, Evaluation
            log_loss = (epoch % log_interval) == 0
            loss_info = self.track_loss(clients_fit_res, log_loss)
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, log_eval=False,
                central_client=server.fed_strategy.name == 'central', other_infos=loss_info, eval = False
            )

            if use_early_stopping:
                self.early_stopping_step(
                    clients_fit_res, early_stoppings, all_clients_converged_sign, early_stopping_mode
                )
                if all(all_clients_converged_sign):
                    loguru.logger.info("All clients have converged. Stopping training at {}.".format(epoch))
                    break

            if epoch == 0 and self.initial_zero_impute == False:
                for client in clients:
                    client.local_imputation(params={})

            if epoch > 0 and epoch % imp_interval == 0:
                for client in clients:
                    client.local_imputation(params={})
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

            #self.pseudo_imp_eval(clients, evaluator)

        ################################################################################################################
        loguru.logger.info("start fine tuning ...")
        ################################################################################################################
        # local training of an imputation model
        early_stoppings, all_clients_converged_sign = self.setup_early_stopping(
            'local', model_converge_tol, model_converge_tolerance_patience,
            model_converge_increase_patience, model_converge_window_size, model_converge_steps,
            model_converge_back_steps, clients
        )

        fine_tune_epochs = server.fed_strategy.fine_tune_epochs
        fit_params_list = [{'local_epoch': 1} for _ in clients]
        for epoch in trange(fine_tune_epochs, desc='Fine Tuning Epoch', colour='blue'):

            clients_fit_res = []
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                client = clients[client_idx]
                fit_params = fit_params_list[client_idx]
                fit_params.update(fit_instruction[client_idx])
                fit_params.update({'freeze_encoder': False})
                # if it is converged, do not fit the model
                if all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                _, fit_res = client.fit_local_imp_model(params=fit_params)
                clients_fit_res.append(fit_res)

            ####################################################################################################
            # Early Stopping and Logging and Evaluation
            log_loss = (epoch % log_interval) == 0
            loss_info = self.track_loss(clients_fit_res, log_loss)
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, log_eval=False,
                central_client=server.fed_strategy.name == 'central', other_infos=loss_info, eval = False
            )

            if epoch % save_model_interval == 0:
                for client in clients:
                    client.save_imp_model(version=f'{epoch + global_model_epochs}')

            if epoch > 0 and epoch % imp_interval == 0:
                for client in clients:
                    client.local_imputation(params={})
                
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch + global_model_epochs,
                    central_client=server.fed_strategy.name == 'central'
                )

            if use_early_stopping:
                self.early_stopping_step(clients_fit_res, early_stoppings, all_clients_converged_sign, 'local')
                if all(all_clients_converged_sign):
                    loguru.logger.info("All clients have converged. Stopping training at {}.".format(epoch))
                    break

        #########################################################################################################
        # Final imputation and Evaluation
        for client in clients:
            client.local_imputation(params={})
            client.save_imp_model(version='final')

        self.eval_and_track(
            evaluator, tracker, clients, phase='final', central_client=server.fed_strategy.name == 'central'
        )

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        ################################################################################################################
        # Initial Imputation and Evaluation
        ################################################################################################################
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        # clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        if self.initial_zero_impute:
            server.fed_strategy.initial_impute = 'zero'
            clients, server = initial_imputation(clients, server)
        else:
            clients, server = initial_imputation(clients, server)
        
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        clients_data = [(client.X_train_imp, client.X_train, client.X_train_mask) for client in clients]
        self.eval_and_track_parallel(
            evaluator, tracker, clients_data, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ################################################################################################################
        # Federated Imputation Workflow
        ################################################################################################################
        use_early_stopping = self.use_early_stopping
        early_stopping_mode = 'local'

        model_converge_tol = self.model_converge['tolerance']
        model_converge_tolerance_patience = self.model_converge['tolerance_patience']
        model_converge_increase_patience = self.model_converge['increase_patience']
        model_converge_window_size = self.model_converge['window_size']
        model_converge_steps = self.model_converge['check_steps']
        model_converge_back_steps = self.model_converge['back_steps']

        early_stoppings, all_clients_converged_sign = self.setup_early_stopping(
            early_stopping_mode, model_converge_tol, model_converge_tolerance_patience,
            model_converge_increase_patience, model_converge_window_size, model_converge_steps,
            model_converge_back_steps, clients
        )

        ################################################################################################################
        # setup clients and server
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
        # Federated Training
        ################################################################################################################
        global_model_epochs = self.global_epoch
        log_interval = self.log_interval
        imp_interval = self.imp_interval
        save_model_interval = self.save_model_interval
        fit_params_list = [{'local_epoch': self.local_epoch} for _ in clients]

        for epoch in trange(global_model_epochs, desc='Global Epoch', colour='blue'):

            # local fitting
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                fit_params = fit_params_list[client_idx]
                fit_params.update(fit_instruction[client_idx])
                if early_stopping_mode == 'local' and all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                pipe = client_pipes[client_idx]
                pipe[0].send(("fit_local", fit_params))

            # receive the results and send to server
            ret = [pipe[0].recv() for pipe in client_pipes]
            local_models = [item[0] for item in ret]
            fit_res_list = [item[1] for item in ret]
            
            # Loss tracking
            log_loss = (epoch % log_interval) == 0
            loss_info = self.track_loss(clients_fit_res, log_loss)
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, log_eval=False,
                central_client=server.fed_strategy.name == 'central', other_infos=loss_info, eval = False
            )

            # server aggregation
            main_pipe.send("aggregate")
            for pipe, params, fit_res in zip(client_pipes, local_models, fit_res_list):
                pipe[1].send((params, fit_res))
            global_models, agg_res = main_pipe.recv()

            # update local model
            for client_id, (pipe, global_model) in enumerate(zip(client_pipes, global_models)):
                if early_stopping_mode == 'local' and (all_clients_converged_sign[client_id]):
                    continue
                pipe[0].send(("update_only", {'global_model_params': global_model, 'params': {}}))
                
            # Server local imputation
            main_pipe.send(("local_impute", {}))
            server.X_test_imp, server.X_test, server.X_test_mask = main_pipe.recv()

            # Early Stopping and Logging and Evaluation
            if use_early_stopping:
                self.early_stopping_step(
                    fit_res_list, early_stoppings, all_clients_converged_sign, early_stopping_mode
                )
                if all(all_clients_converged_sign):
                    loguru.logger.info("All clients have converged. Stopping training at {}.".format(epoch))
                    break

            if epoch % save_model_interval == 0:
                for pipe in client_pipes:
                    pipe[0].send(("save_model", f'{epoch}'))

            if (epoch == 0 and self.initial_zero_impute == False) or (epoch > 0 and epoch % imp_interval == 0):
                for pipe in client_pipes:
                    pipe[0].send(("impute_only", {'params': {}}))

                clients_data = [pipe[0].recv() for pipe in client_pipes]
                self.eval_and_track_parallel(
                    evaluator, tracker, clients_data, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

        ################################################################################################################
        # Federated Fine Tuning
        ################################################################################################################
        loguru.logger.info("start fine tuning ...")
        early_stoppings, all_clients_converged_sign = self.setup_early_stopping(
            'local', model_converge_tol, model_converge_tolerance_patience,
            model_converge_increase_patience, model_converge_window_size, model_converge_steps,
            model_converge_back_steps, clients
        )

        fine_tune_epochs = server.fed_strategy.fine_tune_epochs
        fit_params_list = [{'local_epoch': 1} for _ in clients]
        for epoch in trange(fine_tune_epochs, desc='Fine Tuning Epoch', colour='blue'):

            # local training
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                fit_params = fit_params_list[client_idx]
                fit_params.update(fit_instruction[client_idx])
                fit_params.update({'freeze_encoder': False})
                if all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                pipe = client_pipes[client_idx]
                pipe[0].send(("fit_local", fit_params))

            # receive the results
            clients_fit_res = [pipe[0].recv() for pipe in client_pipes]

            # Early Stopping and Logging and Evaluation
            log_loss = (epoch % log_interval) == 0
            loss_info = self.track_loss(clients_fit_res, log_loss)
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, log_eval=False,
                central_client=server.fed_strategy.name == 'central', other_infos=loss_info, eval = False
            )

            if epoch % save_model_interval == 0:
                for pipe in client_pipes:
                    pipe[0].send(("save_model", f'{epoch + global_model_epochs}'))

            if epoch > 0 and epoch % imp_interval == 0:
                for pipe in client_pipes:
                    pipe[0].send(("impute_only", {'params': {}}))

                clients_data = [pipe[0].recv() for pipe in client_pipes]
                self.eval_and_track_parallel(
                    evaluator, tracker, clients_data, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

            if use_early_stopping:
                self.early_stopping_step(clients_fit_res, early_stoppings, all_clients_converged_sign, 'local')
                if all(all_clients_converged_sign):
                    loguru.logger.info("All clients have converged. Stopping training at {}.".format(epoch))
                    break

        ############################################################################################################
        # Terminate processes and update clients and server and collect environment
        for pipe in client_pipes:
            pipe[0].send(("impute_only", {'params': {}}))
            pipe[0].send(("save_model", 'final'))

        clients_data = [pipe[0].recv() for pipe in client_pipes]
        self.eval_and_track_parallel(
            evaluator, tracker, clients_data, phase='final', central_client=server.fed_strategy.name == 'central'
        )

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
        server.X_test_imp = new_server.X_test_imp
        server.X_test = new_server.X_test
        server.X_test_mask = new_server.X_test_mask

        # Join processes
        for p in client_processes + [server_process]:
            p.join()
            p.close()

        return tracker

    @staticmethod
    def setup_early_stopping(
            early_stopping_mode, model_converge_tol, model_tolerance_patience, model_increase_patience,
            model_converge_window_size,
            model_converge_steps, model_converge_back_steps, clients: List
    ):
        if early_stopping_mode == 'global':
            early_stoppings = [nn_utils.EarlyStopping(
                tolerance_patience=model_tolerance_patience, increase_patience=model_increase_patience,
                tolerance=model_converge_tol, window_size=model_converge_window_size,
                check_steps=model_converge_steps, backward_window_size=model_converge_back_steps
            )]
            all_clients_converged_sign = [False]
        else:
            early_stoppings = [
                nn_utils.EarlyStopping(
                    tolerance_patience=model_tolerance_patience, increase_patience=model_increase_patience,
                    tolerance=model_converge_tol, window_size=model_converge_window_size,
                    check_steps=model_converge_steps, backward_window_size=model_converge_back_steps
                ) for _ in clients
            ]
            all_clients_converged_sign = [False for _ in clients]

        return early_stoppings, all_clients_converged_sign

    @staticmethod
    def early_stopping_step(
            clients_fit_res: List, early_stoppings: List, all_clients_converged_sign: List,
            early_stopping_mode: str = 'global'
    ):

        if early_stopping_mode == 'local':
            for idx, (client_fit_res, early_stopping) in enumerate(zip(clients_fit_res, early_stoppings)):
                if 'loss' not in client_fit_res:
                    continue
                early_stopping.update(client_fit_res['loss'])
                if early_stopping.check_convergence():
                    all_clients_converged_sign[idx] = True
                    loguru.logger.debug(f"Client {idx} has converged.")

        elif early_stopping_mode == 'global':
            avg_loss = np.array(
                [client_fit_res['loss'] for client_fit_res in clients_fit_res if 'loss' in client_fit_res]
            ).mean()
            early_stoppings[0].update(avg_loss)
            if early_stoppings[0].check_convergence():
                all_clients_converged_sign[0] = True

        else:
            raise ValueError(f"Early stopping mode {early_stopping_mode} not supported.")

    @staticmethod
    def logging_loss(clients_fit_res: List):
        losses = np.array([client_fit_res['loss'] for client_fit_res in clients_fit_res if 'loss' in client_fit_res])
        if len(losses) == 0:
            loguru.logger.info("\nLoss: {:.2f} ({:2f})".format(0, 0))
        else:
            loguru.logger.info("\nLoss: {:.4f} ({:4f})".format(losses.mean(), losses.std()))
            
    @staticmethod
    def track_loss(clients_fit_res: List, log: bool = False):
        """
        Track and log the loss of the clients
        """
        losses = np.array([client_fit_res['loss'] for client_fit_res in clients_fit_res if 'loss' in client_fit_res])
        if len(losses) == 0:
            loss_mean, loss_std = 0, 0
        else:
            loss_mean, loss_std = losses.mean(), losses.std()
        
        if log:
            loguru.logger.info("\nLoss: {:.2f} ({:2f})".format(loss_mean, loss_std))
            
        return {
            'loss': {client_idx: client_fit_res['loss'] if 'loss' in client_fit_res else 0 
                     for client_idx, client_fit_res in enumerate(clients_fit_res)}
        }

    @staticmethod
    def pseudo_imp_eval(clients, evaluator: Evaluator):
        X_imps = []
        for client in clients:
            X_imp = client.local_imputation(params={"temp_imp": True})
            X_imps.append(X_imp)

        eval_results = evaluator.evaluate_imputation(
            X_imps, [client.X_train for client in clients], [client.X_train_mask for client in clients]
        )

        loguru.logger.debug(f"Average: {eval_results['imp_rmse_avg']}, {eval_results['imp_ws_avg']}")
        for idx, client in enumerate(clients):
            loguru.logger.debug(
                f"Client {idx}: {eval_results['imp_rmse_clients'][idx]}, {eval_results['imp_ws_clients'][idx]}"
            )
            
    def __str__(self):
        return f"Joint Modeling Workflow"

    def __repr__(self):
        return f"Joint Modeling Workflow"
