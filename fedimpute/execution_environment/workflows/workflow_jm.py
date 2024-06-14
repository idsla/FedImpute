import logging
import loguru
import numpy as np
from tqdm import tqdm

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


class WorkflowJM(BaseWorkflow):

    def __init__(
            self,
            initial_zero_impute:bool = True,
            global_epoch:int = 150,
            local_epoch:int = 5,
            use_early_stopping: bool = True,
            log_interval:int = 10,
            imp_interval:int = 1000,
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
        super().__init__()
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
            clients = initial_imputation('zero', clients)
        else:
            clients = initial_imputation(server.fed_strategy.initial_impute, clients)
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ############################################################################################################
        # Federated Imputation Workflow
        use_early_stopping = self.use_early_stopping
        # if server.fed_strategy.name == 'local':
        #     early_stopping_mode = 'local'
        # else:
        #     early_stopping_mode = 'global'
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

            #############################################################################################
            # Early Stopping, Loss, Evaluation
            if epoch % log_interval == 0:
                self.logging_loss(clients_fit_res)

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
        print(early_stopping_mode, use_early_stopping)
        early_stoppings, all_clients_converged_sign = self.setup_early_stopping(
            'local', model_converge_tol, model_converge_tolerance_patience,
            model_converge_increase_patience, model_converge_window_size, model_converge_steps,
            model_converge_back_steps, clients
        )

        fine_tune_epochs = server.fed_strategy.fine_tune_epochs
        fit_params_list = [{
            'local_epoch': 1
        } for _ in clients]
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
            if epoch % log_interval == 0:
                self.logging_loss(clients_fit_res)

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
            loguru.logger.debug("\nLoss: {:.2f} ({:2f})".format(0, 0))
        else:
            loguru.logger.debug("\nLoss: {:.4f} ({:4f})".format(losses.mean(), losses.std()))

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
