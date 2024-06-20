from abc import ABC, abstractmethod
import loguru
from ..server import Server
from typing import Dict, Union, List, Tuple, Any
from ..client import Client
from ..utils.tracker import Tracker


class BaseWorkflow(ABC):

    """
    Abstract class for the workflow to be used in the federated imputation environment
    """

    def __init__(self):
        pass

    @abstractmethod
    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker
    ) -> Tracker:
        """
        Sequential federated imputation workflow

        Args:
            clients: List[Client] - list of clients
            server: Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results

        Returns:
            Tracker - tracker with tracked results
        """
        pass

    @abstractmethod
    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker
    ) -> Tracker:
        """
        Parallel federated imputation workflow

        Args:
            clients: List[Client] - list of clients
            server: Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results

        Returns:
            Tracker - tracker with tracked results
        """
        pass

    def run_fed_imp(
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker, run_type: str
    ) -> Tracker:

        """
        Run the federated imputation workflow based on the

        Args:
            clients: List[Client] - list of clients
            server:  Server - server
            evaluator: Evaluator - evaluator
            tracker: Tracker - tracker to tracking results
            run_type: str - type of the workflow run (sequential or parallel)

        Returns:
            Tracker - tracker with tracked results
        """

        if run_type == 'sequential':
            return self.fed_imp_sequential(clients, server, evaluator, tracker)
        elif run_type == 'parallel':
            return self.fed_imp_parallel(clients, server, evaluator, tracker)
        else:
            raise ValueError('Invalid workflow run type')

    @staticmethod
    def eval_and_track(
            evaluator, tracker, clients, phase='round', epoch=0, log_eval=True, central_client=True
    ) -> Union[Any]:

        ############################################################################################################
        # Initial evaluation and tracking
        if phase == 'initial':

            evaluation_results = evaluator.evaluate_imputation(
                X_train_imps=[client.X_train_imp for client in clients],
                X_train_origins=[client.X_train for client in clients],
                X_train_masks=[client.X_train_mask for client in clients],
                central_client=central_client
            )

            tracker.record_initial(
                data=[client.X_train for client in clients],
                mask=[client.X_train_mask for client in clients],
                imp_quality=evaluation_results,
            )

            loguru.logger.info(
                f"\nInitial: rmse - {evaluation_results['imp_rmse_avg']} ws - {evaluation_results['imp_ws_avg']}"
            )

            return None

        ############################################################################################################
        # Evaluation and tracking for each round
        elif phase == 'round':

            evaluation_results = evaluator.evaluate_imputation(
                X_train_imps=[client.X_train_imp for client in clients],
                X_train_origins=[client.X_train for client in clients],
                X_train_masks=[client.X_train_mask for client in clients],
                central_client=central_client
            )

            tracker.record_round(
                round_num=epoch + 1,
                imp_quality=evaluation_results,
                data=[client.X_train_imp for client in clients],
                model_params=[],  # todo make it
                other_info=[{} for _ in clients]
            )

            if log_eval:
                loguru.logger.info(
                    f"Epoch {epoch}: rmse - {evaluation_results['imp_rmse_avg']} ws - {evaluation_results['imp_ws_avg']}"
                )

            return evaluator.get_imp_quality(evaluation_results)

        ############################################################################################################
        # Final evaluation and tracking
        elif phase == 'final':

            evaluation_results = evaluator.evaluate_imputation(
                X_train_imps=[client.X_train_imp for client in clients],
                X_train_origins=[client.X_train for client in clients],
                X_train_masks=[client.X_train_mask for client in clients],
                central_client=central_client
            )

            tracker.record_final(
                imp_quality=evaluation_results,
                data=[client.X_train_imp for client in clients],
                model_params=[],
                other_info=[{} for _ in clients]
            )

            loguru.logger.info(
                f"Final: rmse - {evaluation_results['imp_rmse_avg']} ws - {evaluation_results['imp_ws_avg']}"
            )

            return evaluator.get_imp_quality(evaluation_results)
