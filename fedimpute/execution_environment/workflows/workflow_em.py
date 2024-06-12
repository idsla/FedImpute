from collections import OrderedDict
from copy import deepcopy

import loguru
import numpy as np

from .utils import formulate_centralized_client, update_clip_threshold
from .workflow import BaseWorkflow
from src.server import Server
from typing import List
from src.client import Client
from ..evaluation.evaluator import Evaluator

from tqdm.auto import trange
from src.utils.tracker import Tracker
from ..imputation.initial_imputation.initial_imputation import initial_imputation


class WorkflowEM(BaseWorkflow):

    def __init__(
            self,
            workflow_params: dict
    ):
        super().__init__()
        self.workflow_params = workflow_params
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:

        """
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        evaluation_interval = train_params['evaluation_interval']
        max_iterations = train_params['max_iterations']
        save_model_interval = train_params['save_model_interval']

        ############################################################################################################
        # Centralized Initialization
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation and evaluation
        clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )
        # Update Global clip thresholds
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        ########################################################################################################
        # federated EM imputation
        fit_params_list = [train_params.copy() for _ in range(len(clients))]

        # central and local training
        if server.fed_strategy.name == 'central' or server.fed_strategy.name == 'local':
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            local_models, clients_fit_res = [], []
            for client_id in trange(len(clients), desc='Clients', colour='green'):
                client = clients[client_id]
                fit_params = fit_params_list[client_id]
                fit_params['local_epoch'] = max_iterations * train_params['local_epoch']
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

        # federated training
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
                        tolerance=train_params['convergence_thres']
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
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        pass

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
