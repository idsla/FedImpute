from .workflow import BaseWorkflow
from fedimpute.execution_environment.server import Server
from typing import List
import multiprocessing as mp
from fedimpute.execution_environment.client import Client
from fedimpute.execution_environment.imputation.initial_imputation.initial_imputation import initial_imputation
from fedimpute.execution_environment.utils.evaluator import Evaluator
from fedimpute.execution_environment.utils.tracker import Tracker
from .utils import formulate_centralized_client, update_clip_threshold


class WorkflowSimple(BaseWorkflow):

    def __init__(
            self,
    ):
        super().__init__()
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
        clients = initial_imputation(server.fed_strategy.initial_impute, clients)

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

        """
        Imputation workflow for MICE Multiprocessing Version
        """

        # Setup multiprocessing
        mp.set_start_method('spawn')
        client_pipes = [mp.Pipe() for _ in clients]
        server_pipe, main_pipe = mp.Pipe()
        evaluator_pipe, eval_main_pipe = mp.Pipe()
        tracker_pipe, track_main_pipe = mp.Pipe()

        # Start processes
        client_processes = [mp.Process(target=client_process, args=(client, pipe[1])) for client, pipe in
                            zip(clients, client_pipes)]
        server_process = mp.Process(target=server_process,
                                    args=(server, [pipe[0] for pipe in client_pipes], server_pipe))

        evaluator_process = mp.Process(target=evaluator_process, args=(evaluator, evaluator_pipe, tracker_pipe))
        tracker_process = mp.Process(target=tracker_process, args=(tracker, tracker_pipe))

        for p in client_processes + [server_process, evaluator_process, tracker_process]:
            p.start()

        # Workflow Parameters
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))
            # Add an additional pipe for the central client
            central_pipe, central_client_pipe = mp.Pipe()
            client_pipes.append((central_pipe, central_client_pipe))
            central_process = mp.Process(target=client_process, args=(clients[-1], central_client_pipe))
            central_process.start()

        # Initial Imputation
        for pipe, client in zip(client_pipes, clients):
            pipe[0].send(("initial_impute", server.fed_strategy.initial_impute))

        # Initial evaluation and tracking
        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        # Federated imputation
        fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
        for pipe, client in zip(client_pipes, clients):
            train_params = {}
            train_params.update(fit_instruction[client.client_id])
            pipe[0].send(("fit_local", train_params))

        main_pipe.send("aggregate")
        global_models = main_pipe.recv()

        for pipe, global_model in zip(client_pipes, global_models):
            pipe[0].send(("update_and_impute", {'global_model': global_model, 'params': {}}))

        # Final Evaluation and Tracking and saving imputation model
        self.eval_and_track(
            evaluator, tracker, clients, phase='final', central_client=server.fed_strategy.name == 'central'
        )

        for pipe in client_pipes:
            pipe[0].send(("save_model", None))

        # Terminate processes
        for pipe in client_pipes:
            pipe[0].send(("terminate", None))
        main_pipe.send("terminate")

        # Get final results from tracker
        track_main_pipe.send(("get_results", None))
        final_results = track_main_pipe.recv()
        track_main_pipe.send(("terminate", None))

        # Join processes
        for p in client_processes + [server_process, evaluator_process, tracker_process]:
            p.join()

        if server.fed_strategy.name == 'central':
            central_process.join()

        return final_results


def client_process(client, client_pipe):
    while True:
        command, data = client_pipe.recv()
        if command == "initial_impute":
            client.initial_impute(data)
        elif command == "fit_local":
            params, fit_res = client.fit_local_imp_model(data)
            client_pipe.send((params, fit_res))
        elif command == "update_and_impute":
            client.update_local_imp_model(data['global_model'], params=data['params'])
            client.local_imputation(params=data['params'])
        elif command == "save_model":
            client.save_imp_model(version='final')
        elif command == "terminate":
            break

def server_process(server, client_pipes, server_pipe):
    while True:
        command = server_pipe.recv()
        if command == "aggregate":
            params_list, fit_rest_list = [], []
            for pipe in client_pipes:
                params, fit_res = pipe.recv()
                params_list.append(params)
                fit_rest_list.append(fit_res)
            global_models, agg_res = server.fed_strategy.aggregate_parameters(params_list, fit_rest_list, {})
            server_pipe.send(global_models)
        elif command == "terminate":
            break


def evaluator_process(evaluator, evaluator_pipe, tracker_pipe):
    while True:
        command, data = evaluator_pipe.recv()
        if command == "evaluate":
            eval_results = evaluator.evaluate(
                data["client_id"],
                data["imputed_data"],
                data["original_data"],
                data["mask"]
            )
            tracker_pipe.send(("update", {
                "client_id": data["client_id"],
                "phase": data["phase"],
                "results": eval_results
            }))
        elif command == "terminate":
            break

def tracker_process(tracker, tracker_pipe):
    while True:
        command, data = tracker_pipe.recv()
        if command == "update":
            tracker.update(data["client_id"], data["phase"], data["results"])
        elif command == "get_results":
            tracker_pipe.send(tracker.get_results())
        elif command == "terminate":
            break