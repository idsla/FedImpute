from .workflow import BaseWorkflow
from fedimpute.execution_environment.server import Server
from typing import List
import multiprocessing as mp
from fedimpute.execution_environment.client import Client
from fedimpute.execution_environment.imputation.initial_imputation.initial_imputation import initial_imputation
from fedimpute.execution_environment.utils.evaluator import Evaluator
from fedimpute.execution_environment.utils.tracker import Tracker
from .utils import formulate_centralized_client, update_clip_threshold
from .parallel import client_process_func, server_process_func


class WorkflowSimple(BaseWorkflow):

    def __init__(
            self,
    ):
        super().__init__('MEAN (Mean Imputation)')
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        """
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation
        clients, server = initial_imputation(clients, server)

        # initial evaluation and tracking
        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ############################################################################################################
        # federated imputation
        params_list, fit_rest_list = [], []
        fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
        for client in clients:
            train_params = {}
            train_params.update(fit_instruction[client.client_id])
            params, fit_res = client.fit_local_imp_model(train_params)
            params_list.append(params)
            fit_rest_list.append(fit_res)

        global_models, agg_res = server.fed_strategy.aggregate_parameters(params_list, fit_rest_list, {})

        for global_model, client in zip(global_models, clients):
            client.update_local_imp_model(global_model, params={})
            client.local_imputation(params={})
            
        server.local_imputation(params={})

        ########################################################################################################
        # Final Evaluation and Tracking and saving imputation model
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
        # Initial Imputation and setup
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        clients, server = initial_imputation(clients, server)
        clients_data = [(client.X_train_imp, client.X_train, client.X_train_mask) for client in clients]
        self.eval_and_track_parallel(
            evaluator, tracker, clients_data, phase='initial', central_client=server.fed_strategy.name == 'central'
        )
        
        ############################################################################################################
        # Server and Client setup

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

        ############################################################################################################
        # Federated imputation
        # Client fit local model
        fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
        for pipe, client in zip(client_pipes, clients):
            train_params = {}
            train_params.update(fit_instruction[client.client_id])
            pipe[0].send(("fit_local", train_params))

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
            
        # Server local imputation
        main_pipe.send(("local_impute", {}))
        server.X_test_imp, server.X_test, server.X_test_mask = main_pipe.recv()

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
        server.X_test_imp = new_server.X_test_imp
        server.X_test = new_server.X_test
        server.X_test_mask = new_server.X_test_mask

        # Join processes
        for p in client_processes + [server_process]:
            p.join()
            p.close()

        return tracker
    
    def __str__(self):
        return f"Simple Imputation Workflow"

    def __repr__(self):
        return f"Simple Imputation Workflow"
