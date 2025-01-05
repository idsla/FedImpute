from abc import ABC, abstractmethod
import loguru
import timeit
import sys

from ..server import Server
from typing import Dict, Union, List, Tuple, Any
from ..client import Client
from ..utils.tracker import Tracker

class BaseWorkflow(ABC):

    """
    Abstract class for the workflow to be used in the federated imputation environment
    """

    def __init__(self, name: str):
        self.name = name

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
            self, clients: List[Client], server: Server, evaluator, tracker: Tracker, run_type: str, verbose: int
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
        # Set logging level based on verbosity
        loguru.logger.remove()  # Remove default handler
        if verbose == 0:
            pass
        elif verbose == 1:
            loguru.logger.add(
                sys.stdout, 
                level="SUCCESS",
                format="<level>{message}</level>"
            )
        elif verbose == 2:
            loguru.logger.add(
                sys.stdout, 
                level="INFO",
                format="<level>{message}</level>"
            ) 
        elif verbose >= 3:
            loguru.logger.add(
                sys.stdout, 
                level="DEBUG",
            )

        loguru.logger.success(f"Imputation Start ...")
        if run_type == 'sequential':
            start_time = timeit.default_timer()
            result = self.fed_imp_sequential(clients, server, evaluator, tracker)
            end_time = timeit.default_timer()
            loguru.logger.success(f"Finished. Running time: {end_time - start_time:.4f} seconds")
            return result
        elif run_type == 'parallel':
            start_time = timeit.default_timer()
            result = self.fed_imp_parallel(clients, server, evaluator, tracker)
            end_time = timeit.default_timer()
            loguru.logger.success(f"Finished. Running time: {end_time - start_time:.4f} seconds")
            return result
        else:
            raise ValueError('Invalid workflow run type')

    @staticmethod
    def eval_and_track(
        evaluator, 
        tracker, 
        clients, 
        phase='round', 
        epoch=0, 
        log_eval=True, 
        central_client=True, 
        other_infos=None,
        eval = True
    ) -> Union[Any]:

        ############################################################################################################
        # Initial evaluation and tracking
        if phase == 'initial':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_initial(
                    data=[client.X_train for client in clients],
                    mask=[client.X_train_mask for client in clients],
                    imp_quality=None,
                )
                if log_eval:
                    loguru.logger.info("Initial Imputation.")
                
                return None
                
            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial Imputation."
                        )
                
                tracker.record_initial(
                    data=[client.X_train for client in clients],
                    mask=[client.X_train_mask for client in clients],
                    imp_quality=evaluation_results,
                )

                return None

        ############################################################################################################
        # Evaluation and tracking for each round
        elif phase == 'round':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=None,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],  # todo make it
                    other_info=[other_infos[client_id] for client_id in range(len(clients))]
                )
                
                avg_other_info = {}
                for key in other_infos[0].keys():
                    avg_other_info[key] = sum(info[key] for info in other_infos) / len(other_infos)
                
                if log_eval:
                    info_str = f"Epoch {epoch} "
                    for key in avg_other_info.keys():
                        info_str += f"{key}: {avg_other_info[key]:.4f} "
                
                    loguru.logger.info(info_str)
                
                return None
            
            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client,
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch}: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch} ..."
                        )

                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=evaluation_results,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],  # todo make it
                    other_info=[other_infos[client_id] for client_id in range(len(clients))]
                )

                return evaluator.get_imp_quality(evaluation_results)

        ############################################################################################################
        # Final evaluation and tracking
        elif phase == 'final':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_final(
                    imp_quality=None,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],
                    other_info=None
                )
                
                if log_eval:
                    loguru.logger.info(f"Final Round ...")
                
                return None
                
            else:
                
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=[client.X_train_imp for client in clients],
                        X_train_origins=[client.X_train for client in clients],
                        X_train_masks=[client.X_train_mask for client in clients],
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final Round ..."
                        )

                tracker.record_final(
                    imp_quality=evaluation_results,
                    data=[client.X_train_imp for client in clients],
                    model_params=[],
                    other_info=None
                )
                
                return evaluator.get_imp_quality(evaluation_results)

    @staticmethod
    def eval_and_track_parallel(
        evaluator, tracker, clients_data, phase='round', epoch=0, log_eval=True, central_client=True, other_infos=None
    ):

        ############################################################################################################
        X_train_imps, X_train_origins, X_train_masks = [], [], []
        for X_train_imp, X_train_origin, X_train_mask in clients_data:
            X_train_imps.append(X_train_imp)
            X_train_origins.append(X_train_origin)
            X_train_masks.append(X_train_mask)

        ############################################################################################################
        # Initial evaluation and tracking
        if phase == 'initial':
            
            if any(client.no_ground_truth for client in clients):
                tracker.record_initial(
                    data=X_train_origins, mask=X_train_masks, imp_quality=None,
                )

                if log_eval:
                    loguru.logger.info("Initial Imputation.")
                
            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins, X_train_masks=X_train_masks,
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Initial Imputation."
                        )

                tracker.record_initial(
                    data=X_train_origins, mask=X_train_masks, imp_quality=evaluation_results,
                )

            return None

        ############################################################################################################
        # Evaluation and tracking for each round
        elif phase == 'round':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=None,
                    data=X_train_imps,
                    model_params=[],
                    other_info=[other_infos[client_id] for client_id in range(len(X_train_imps))]
                )
                
                avg_other_info = {}
                for key in other_infos[0].keys():
                    avg_other_info[key] = sum(info[key] for info in other_infos) / len(other_infos)
                
                info_str = f"Epoch {epoch} "
                for key in avg_other_info.keys():
                    info_str += f"{key}: {avg_other_info[key]:.4f} "
                
                loguru.logger.info(info_str)
                
                return None

            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins, X_train_masks=X_train_masks,
                        central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch}: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Epoch {epoch} ..."
                        )

                tracker.record_round(
                    round_num=epoch + 1,
                    imp_quality=evaluation_results,
                    data=X_train_imps,
                    model_params=[],  # todo make it
                    other_info=[other_infos[client_id] for client_id in range(len(X_train_imps))]
                )

                return evaluator.get_imp_quality(evaluation_results)

        ############################################################################################################
        # Final evaluation and tracking
        elif phase == 'final':
            
            if any(client.no_ground_truth for client in clients):
                
                tracker.record_final(
                    imp_quality=None,
                    data=X_train_imps,
                    model_params=[],
                    other_info=None
                )

                if log_eval:
                    loguru.logger.info(f"Final Round ...")
                
                return None

            else:
                if eval:
                    evaluation_results = evaluator.evaluate_imputation(
                        X_train_imps=X_train_imps, X_train_origins=X_train_origins,
                        X_train_masks=X_train_masks, central_client=central_client
                    )
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final: rmse - {evaluation_results['imp_rmse_avg']:.4f} "
                            f"ws - {evaluation_results['imp_ws_avg']:.4f}"
                        )
                    
                else:
                    evaluation_results = None
                    
                    if log_eval:
                        loguru.logger.info(
                            f"Final Round ..."
                        )

                tracker.record_final(
                    imp_quality=evaluation_results, data=X_train_imps, model_params=[],
                    other_info=None
                )

                return evaluator.get_imp_quality(evaluation_results)
